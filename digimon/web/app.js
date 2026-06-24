// Digimon RPG Engine - Enhanced for Multiple Series Selection

let player = null;
let enemy = null;
let isPlayerTurn = true;
let isBattling = false;
let canEvolve = false;
let evolutionTimer = null;
let inventory = { "HP Potion": 3 };
let isInventoryOpen = false;

// Audio System
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
let soundEnabled = true;
let bgmOsc = null;
let bgmGain = null;
let bgmInterval = null;

function toggleSound() {
    soundEnabled = document.getElementById('sound-toggle').checked;
    if (soundEnabled && audioCtx.state === 'suspended') {
        audioCtx.resume();
    } else if (!soundEnabled) {
        stopBGM();
    }
}

document.getElementById('sound-toggle').addEventListener('change', toggleSound);

function playSound(type) {
    if (!soundEnabled) return;
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    const now = audioCtx.currentTime;
    
    if (type === 'hit') {
        osc.type = 'square';
        osc.frequency.setValueAtTime(150, now);
        osc.frequency.exponentialRampToValueAtTime(40, now + 0.1);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
        osc.start(now);
        osc.stop(now + 0.1);
    } else if (type === 'attack') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(300, now);
        osc.frequency.linearRampToValueAtTime(600, now + 0.1);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.linearRampToValueAtTime(0.01, now + 0.1);
        osc.start(now);
        osc.stop(now + 0.1);
    } else if (type === 'levelUp') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(400, now);
        osc.frequency.setValueAtTime(500, now + 0.1);
        osc.frequency.setValueAtTime(600, now + 0.2);
        osc.frequency.setValueAtTime(800, now + 0.3);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.linearRampToValueAtTime(0, now + 0.5);
        osc.start(now);
        osc.stop(now + 0.5);
    } else if (type === 'item') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(600, now);
        osc.frequency.exponentialRampToValueAtTime(1200, now + 0.2);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.linearRampToValueAtTime(0, now + 0.2);
        osc.start(now);
        osc.stop(now + 0.2);
    }
}

function startBGM(type) {
    if (!soundEnabled) return;
    stopBGM();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    
    bgmOsc = audioCtx.createOscillator();
    bgmGain = audioCtx.createGain();
    bgmOsc.connect(bgmGain);
    bgmGain.connect(audioCtx.destination);
    
    bgmGain.gain.value = 0.02;
    bgmOsc.type = 'triangle';
    
    const notesBattle = [200, 250, 300, 250];
    const notesRPG = [300, 400, 350, 450, 500, 400];
    let notes = type === 'battle' ? notesBattle : notesRPG;
    let idx = 0;
    
    bgmOsc.start();
    
    bgmInterval = setInterval(() => {
        if (!soundEnabled || !bgmOsc) return;
        bgmOsc.frequency.setValueAtTime(notes[idx], audioCtx.currentTime);
        idx = (idx + 1) % notes.length;
    }, type === 'battle' ? 200 : 400);
}

function stopBGM() {
    if (bgmInterval) clearInterval(bgmInterval);
    if (bgmOsc) {
        bgmOsc.stop();
        bgmOsc.disconnect();
        bgmOsc = null;
    }
    if (bgmGain) {
        bgmGain.disconnect();
        bgmGain = null;
    }
}

// RPG State
let currentMapId = "file_island";
let mapData = null;
let pX = 0;
let pY = 0;
const TILE_SIZE = 20;

// DOM Elements
const screens = {
    title: document.getElementById('screen-title'),
    select: document.getElementById('screen-select'),
    rpg: document.getElementById('screen-rpg'),
    battle: document.getElementById('screen-battle'),
    end: document.getElementById('screen-end')
};

const canvas = document.getElementById('rpg-canvas');
const ctx = canvas.getContext('2d');

function switchScreen(name) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[name].classList.add('active');
    
    if (name === 'rpg') startBGM('rpg');
    else if (name === 'battle') startBGM('battle');
    else stopBGM();
}

// Save & Load
if (localStorage.getItem('digimon_save')) {
    document.getElementById('btn-load').style.display = 'inline-block';
}

document.getElementById('btn-start').onclick = initSelection;
document.getElementById('btn-load').onclick = loadGame;
document.getElementById('btn-save').onclick = saveGame;

function saveGame() {
    const saveData = {
        player,
        inventory,
        currentMapId,
        pX,
        pY,
        mapEvents: JSON.parse(JSON.stringify(MAPS[currentMapId].events))
    };
    localStorage.setItem('digimon_save', JSON.stringify(saveData));
    alert("Game Saved!");
    playSound('item');
    document.getElementById('btn-load').style.display = 'inline-block';
}

function loadGame() {
    const dataStr = localStorage.getItem('digimon_save');
    if (!dataStr) return;
    const data = JSON.parse(dataStr);
    player = data.player;
    inventory = data.inventory || {};
    currentMapId = data.currentMapId;
    pX = data.pX;
    pY = data.pY;
    if (data.mapEvents) MAPS[currentMapId].events = data.mapEvents;
    
    if (audioCtx.state === 'suspended') audioCtx.resume();
    loadMap(currentMapId, true);
}

function initSelection() {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const container = document.getElementById('selection-container');
    container.innerHTML = '';
    
    // Group rookies by series
    const seriesList = ["Adventure", "02", "Tamers", "Frontier"];
    seriesList.forEach(seriesName => {
        const header = document.createElement('h3');
        header.style.width = '100%';
        header.style.fontSize = '10px';
        header.style.marginTop = '10px';
        header.style.color = '#0f380f';
        header.innerText = `--- ${seriesName} ---`;
        container.appendChild(header);

        Object.keys(DIGIMON_DATA).forEach(name => {
            const d = DIGIMON_DATA[name];
            if (d.series === seriesName && (d.level === "Rookie" || d.level === "Human Spirit")) {
                const card = document.createElement('div');
                card.className = 'digimon-card';
                card.innerHTML = `<img src="${d.img}"><h3>${name}</h3>`;
                card.onclick = () => selectPartner({ ...d, name: name });
                container.appendChild(card);
            }
        });
    });
    
    switchScreen('select');
}

function selectPartner(partner) {
    player = { 
        ...partner, 
        baseHp: partner.hp, maxHp: partner.hp, currentHp: partner.hp,
        baseAtk: partner.atk, atk: partner.atk,
        baseDef: partner.def, def: partner.def,
        levelNum: 1, exp: 0, maxExp: 100,
        species: partner.name
    };
    inventory = { "HP Potion": 3 };
    loadMap("file_island");
}

function gainExp(amount) {
    player.exp += amount;
    let leveledUp = false;
    while (player.exp >= player.maxExp) {
        player.exp -= player.maxExp;
        player.levelNum++;
        player.maxExp = Math.floor(player.maxExp * 1.5);
        
        // Boost stats 15%
        player.maxHp = Math.floor(player.maxHp * 1.15);
        player.atk = Math.floor(player.atk * 1.15);
        player.def = Math.floor(player.def * 1.15);
        player.currentHp = player.maxHp;
        leveledUp = true;
    }
    if (leveledUp) {
        logMessage(`${player.name} leveled up to Lv.${player.levelNum}! Stats increased!`);
        playSound('levelUp');
    } else {
        logMessage(`Gained ${amount} EXP!`);
    }
    updateMiniStatus();
}

function updateMiniStatus() {
    document.getElementById('mini-sprite').src = player.img;
    document.getElementById('mini-name').innerText = player.name;
    document.getElementById('mini-level').innerText = `Lv. ${player.levelNum}`;
    document.getElementById('mini-hp').innerText = `HP: ${Math.floor(player.currentHp)}/${player.maxHp}`;
    document.getElementById('mini-exp').innerText = `EXP: ${player.exp}/${player.maxExp}`;
    
    if (screens.battle.classList.contains('active')) {
        document.getElementById('battle-player-level').innerText = `Lv.${player.levelNum}`;
    }
}

// === INVENTORY SYSTEM ===

function openInventory(context) {
    isInventoryOpen = true;
    const modal = document.getElementById('inventory-modal');
    const list = document.getElementById('inventory-list');
    const msg = document.getElementById('inv-empty-msg');
    modal.style.display = 'block';
    list.innerHTML = '';
    
    let hasItems = false;
    for (let [itemName, qty] of Object.entries(inventory)) {
        if (qty > 0) {
            hasItems = true;
            const itemData = ITEMS[itemName];
            const div = document.createElement('div');
            div.style.display = 'flex';
            div.style.justifyContent = 'space-between';
            div.style.alignItems = 'center';
            div.style.background = '#8bac0f';
            div.style.padding = '5px';
            div.style.borderRadius = '3px';
            
            div.innerHTML = `
                <div><strong>${itemName} x${qty}</strong><br><span style="font-size:6px;">${itemData.desc}</span></div>
            `;
            const btn = document.createElement('button');
            btn.className = 'btn';
            btn.innerText = 'Use';
            btn.style.fontSize = '6px';
            btn.style.padding = '5px';
            btn.onclick = () => useItem(itemName, context);
            div.appendChild(btn);
            list.appendChild(div);
        }
    }
    
    msg.style.display = hasItems ? 'none' : 'block';
}

function closeInventory() {
    isInventoryOpen = false;
    document.getElementById('inventory-modal').style.display = 'none';
}

document.getElementById('btn-close-inv').onclick = closeInventory;
document.getElementById('btn-inv-rpg').onclick = () => openInventory('rpg');
document.getElementById('btn-inv-battle').onclick = () => {
    if (!isPlayerTurn || !isBattling) return;
    openInventory('battle');
};

function useItem(itemName, context) {
    if (inventory[itemName] <= 0) return;
    const item = ITEMS[itemName];
    
    inventory[itemName]--;
    playSound('item');
    
    let effectMsg = "";
    if (item.type === "heal") {
        player.currentHp = Math.min(player.maxHp, player.currentHp + item.value);
        effectMsg = `Recovered ${item.value} HP!`;
    } else if (item.type === "buff") {
        player.atk += item.value;
        effectMsg = `ATK increased by ${item.value}!`;
    }
    
    closeInventory();
    updateMiniStatus();
    
    if (context === 'battle') {
        logMessage(`Used ${itemName}! ${effectMsg}`);
        updateHealthBar('player', player.currentHp, player.maxHp);
        isPlayerTurn = false;
        setTimeout(enemyTurn, 1000);
    } else {
        alert(`Used ${itemName}! ${effectMsg}`);
    }
}

// === RPG ENGINE ===

const playerImg = new Image();
function loadMap(mapId, isLoad = false) {
    currentMapId = mapId;
    mapData = MAPS[mapId];
    if (!isLoad) {
        pX = mapData.spawnX;
        pY = mapData.spawnY;
    }
    
    document.getElementById('map-name').innerText = mapData.name;
    playerImg.src = player.img;
    updateMiniStatus();
    switchScreen('rpg');
    drawMap();
}

function drawMap() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let y = 0; y < mapData.height; y++) {
        for (let x = 0; x < mapData.width; x++) {
            let tile = mapData.grid[y][x];
            
            if (tile === 0) ctx.fillStyle = "#8bac0f"; // Grass
            else if (tile === 1) ctx.fillStyle = "#306230"; // Wall
            else if (tile === 2) ctx.fillStyle = "#0f380f"; // Water
            else if (tile === 3) ctx.fillStyle = "#c0392b"; // Boss
            else if (tile === 4) ctx.fillStyle = "#2980b9"; // Portal
            else if (tile === 5) ctx.fillStyle = "#9bbc0f"; // Road
            else if (tile === 6) ctx.fillStyle = "#f39c12"; // Chest
            
            ctx.fillRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            ctx.strokeStyle = "rgba(15, 56, 15, 0.1)";
            ctx.strokeRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
        }
    }
    
    // Draw Player
    ctx.drawImage(playerImg, pX * TILE_SIZE, pY * TILE_SIZE, TILE_SIZE, TILE_SIZE);
}

function movePlayer(dx, dy) {
    if (!screens.rpg.classList.contains('active') || isInventoryOpen) return;
    
    let nx = pX + dx;
    let ny = pY + dy;
    
    if (nx >= 0 && nx < mapData.width && ny >= 0 && ny < mapData.height) {
        let tile = mapData.grid[ny][nx];
        if (tile !== 1 && tile !== 2) {
            pX = nx;
            pY = ny;
            drawMap();
            checkTileEvent(nx, ny);
        }
    }
}

function checkTileEvent(x, y) {
    const key = `${x},${y}`;
    const ev = mapData.events && mapData.events[key];
    
    if (ev) {
        if (ev.type === "portal") {
            alert(ev.msg);
            loadMap(ev.target);
            return;
        } else if (ev.type === "boss") {
            startBattle(ev.enemy, true, {x, y, key});
            return;
        } else if (ev.type === "chest") {
            playSound('item');
            alert(ev.msg);
            inventory[ev.item] = (inventory[ev.item] || 0) + 1;
            mapData.grid[y][x] = 0;
            delete mapData.events[key];
            drawMap();
            return;
        }
    }
    
    // Random Encounter
    if ((mapData.grid[y][x] === 0 || mapData.grid[y][x] === 5) && Math.random() < 0.10) {
        const randEnemy = mapData.encounters[Math.floor(Math.random() * mapData.encounters.length)];
        startBattle(randEnemy, false);
    }
}

// === BATTLE ENGINE ===

let currentBattleEvent = null;
function startBattle(enemyName, isBoss, eventRef = null) {
    const eData = ENEMIES[enemyName];
    currentBattleEvent = eventRef;
    
    let enemyScale = isBoss ? 1 : 1 + (player.levelNum * 0.1);
    
    enemy = { 
        ...eData, 
        name: enemyName, 
        maxHp: Math.floor(eData.hp * enemyScale), 
        currentHp: Math.floor(eData.hp * enemyScale),
        atk: Math.floor(eData.atk * enemyScale),
        def: Math.floor(eData.def * enemyScale),
        isBoss: isBoss 
    };
    
    setupBattleUI();
    logMessage(`Encountered ${enemy.name}!`);
    isPlayerTurn = true;
    isBattling = true;
    switchScreen('battle');
    
    startEvolutionTimer();
}

function startEvolutionTimer() {
    canEvolve = false;
    document.getElementById('btn-evolve').style.display = 'none';
    if (evolutionTimer) clearTimeout(evolutionTimer);
    
    let timer = player.level === "Rookie" || player.level === "Human Spirit" ? 6000 : 12000;
    
    evolutionTimer = setTimeout(() => {
        if (isBattling && player.evolve) {
            canEvolve = true;
            document.getElementById('btn-evolve').style.display = 'block';
            logMessage("POWER SURGING! READY TO EVOLVE!");
            playSound('levelUp');
        }
    }, timer);
}

document.getElementById('btn-evolve').onclick = async () => {
    if (!canEvolve) return;
    canEvolve = false;
    document.getElementById('btn-evolve').style.display = 'none';
    
    const evolvedName = player.evolve;
    logMessage(`${player.name} is evolving into ${evolvedName}!!`);
    playSound('levelUp');
    
    document.getElementById('screen').classList.add('shake');
    await new Promise(r => setTimeout(r, 1000));
    document.getElementById('screen').classList.remove('shake');
    
    const evoData = DIGIMON_DATA[evolvedName];
    player.species = evolvedName;
    player.level = evoData.level;
    player.name = evolvedName;
    player.img = evoData.img;
    player.skills = evoData.skills;
    player.evolve = evoData.evolve;
    
    let scale = Math.pow(1.15, player.levelNum - 1);
    let newMaxHp = Math.floor(evoData.hp * scale);
    player.currentHp = player.currentHp + (newMaxHp - player.maxHp);
    player.maxHp = newMaxHp;
    player.atk = Math.floor(evoData.atk * scale);
    player.def = Math.floor(evoData.def * scale);
    
    if (player.currentHp > player.maxHp) player.currentHp = player.maxHp;
    
    playerImg.src = player.img;
    updateMiniStatus();
    setupBattleUI();
    logMessage(`${player.name} ready for combat!`);
    
    if (player.evolve) startEvolutionTimer();
};

function setupBattleUI() {
    document.getElementById('player-name').innerText = player.name;
    document.getElementById('player-img').src = player.img;
    document.getElementById('battle-player-level').innerText = `Lv.${player.levelNum}`;
    updateHealthBar('player', player.currentHp, player.maxHp);

    document.getElementById('enemy-name').innerText = enemy.name;
    document.getElementById('enemy-level').innerText = enemy.isBoss ? "Lv.MAX" : `Lv.${player.levelNum}`;
    document.getElementById('enemy-img').src = enemy.img;
    updateHealthBar('enemy', enemy.currentHp, enemy.maxHp);

    const actions = document.querySelector('.actions');
    actions.innerHTML = '';
    player.skills.forEach(skill => {
        const btn = document.createElement('button');
        btn.className = 'btn action-btn';
        btn.innerText = skill.name;
        btn.onclick = () => playerAttack(skill);
        actions.appendChild(btn);
    });
}

function updateHealthBar(target, hp, maxHp) {
    const percent = Math.max(0, (hp / maxHp) * 100);
    document.getElementById(`${target}-hp-bar`).style.width = `${percent}%`;
    document.getElementById(`${target}-hp-text`).innerText = `${Math.max(0, Math.floor(hp))}/${maxHp}`;
}

async function playerAttack(skill) {
    if (!isPlayerTurn || !isBattling || isInventoryOpen) return;
    isPlayerTurn = false;
    
    playSound('attack');
    await performAttack(player, skill, enemy, 'enemy');
    if (enemy.currentHp <= 0) return winBattle();
    
    setTimeout(enemyTurn, 1000);
}

async function enemyTurn() {
    if (!isBattling) return;
    const skill = enemy.skills[Math.floor(Math.random() * enemy.skills.length)];
    
    playSound('attack');
    await performAttack(enemy, skill, player, 'player');
    
    if (player.currentHp <= 0) return loseBattle();
    
    updateMiniStatus();
    isPlayerTurn = true;
    logMessage("Select your move!");
}

async function performAttack(attacker, skill, defender, targetStr) {
    let dmg = (attacker.atk * (skill.power / 50)) - (defender.def / 2);
    dmg = Math.max(15, Math.floor(dmg * (0.8 + Math.random() * 0.4)));
    
    defender.currentHp -= dmg;
    logMessage(`${attacker.name} uses ${skill.name}! (${dmg} DMG)`);
    
    const atkImg = document.getElementById(`${attacker === player ? 'player' : 'enemy'}-img`);
    const defImg = document.getElementById(`${targetStr}-img`);
    
    atkImg.classList.add(attacker === player ? 'attack-anim' : 'attack-anim-enemy');
    setTimeout(() => {
        playSound('hit');
        atkImg.classList.remove('attack-anim', 'attack-anim-enemy');
        document.getElementById('screen').classList.add('shake');
        defImg.classList.add('hit');
        updateHealthBar(targetStr, defender.currentHp, defender.maxHp);
        setTimeout(() => {
            document.getElementById('screen').classList.remove('shake');
            defImg.classList.remove('hit');
        }, 300);
    }, 200);

    return new Promise(r => setTimeout(r, 1000));
}

async function winBattle() {
    isBattling = false;
    if (evolutionTimer) clearTimeout(evolutionTimer);
    logMessage(`${enemy.name} was defeated!`);
    
    if (currentBattleEvent) {
        mapData.grid[currentBattleEvent.y][currentBattleEvent.x] = 0;
        delete mapData.events[currentBattleEvent.key];
    }
    
    player.currentHp = Math.min(player.maxHp, player.currentHp + Math.floor(player.maxHp * (enemy.isBoss ? 0.5 : 0.2)));
    
    if (enemy.drop && Math.random() < 0.5) {
        inventory[enemy.drop] = (inventory[enemy.drop] || 0) + 1;
        logMessage(`Found ${enemy.drop}!`);
        playSound('item');
        await new Promise(r => setTimeout(r, 1000));
    }
    
    gainExp(enemy.exp_yield);
    
    await new Promise(r => setTimeout(r, 1500));
    
    if (enemy.name === "Lucemon") {
        stopBGM();
        document.getElementById('end-title').innerText = "LEGENDARY TRIUMPH!";
        document.getElementById('end-message').innerText = "All digital worlds are united in peace. You are the true Digimon Master!";
        switchScreen('end');
    } else {
        switchScreen('rpg');
        drawMap();
    }
}

function loseBattle() {
    isBattling = false;
    if (evolutionTimer) clearTimeout(evolutionTimer);
    stopBGM();
    switchScreen('end');
    document.getElementById('end-title').innerText = "JOURNEY ENDS";
    document.getElementById('end-message').innerText = "The Digital World has been reformatted...";
    const btn = document.getElementById('btn-continue');
    btn.innerText = "Restart Game";
    btn.onclick = () => location.reload();
}

function logMessage(msg) {
    document.getElementById('battle-log').innerHTML = `<p>${msg}</p>`;
    document.getElementById('battle-log').scrollTop = document.getElementById('battle-log').scrollHeight;
}

// Input Handling
window.addEventListener('keydown', (e) => {
    if (e.key === "ArrowUp" || e.key === "w") movePlayer(0, -1);
    if (e.key === "ArrowDown" || e.key === "s") movePlayer(0, 1);
    if (e.key === "ArrowLeft" || e.key === "a") movePlayer(-1, 0);
    if (e.key === "ArrowRight" || e.key === "d") movePlayer(1, 0);
});

// Touch D-Pad
document.getElementById('d-up').onclick = () => movePlayer(0, -1);
document.getElementById('d-down').onclick = () => movePlayer(0, 1);
document.getElementById('d-left').onclick = () => movePlayer(-1, 0);
document.getElementById('d-right').onclick = () => movePlayer(1, 0);
