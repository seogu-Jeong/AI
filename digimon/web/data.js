// Digimon Adventure, 02, Tamers, Frontier - Mega Database (Fixed Images v2)

const ITEMS = {
    "HP Potion": { type: "heal", value: 100, desc: "Heals 100 HP" },
    "Mega Potion": { type: "heal", value: 500, desc: "Heals 500 HP" },
    "Evo Shard": { type: "buff", value: 20, desc: "Increases ATK slightly (Battle)" }
};

const DIGIMON_DATA = {
    // --- Adventure (Classic) ---
    "Agumon": { series: "Adventure", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Agumon.png", hp: 150, atk: 40, def: 25, skills: [{ name: "Pepper Breath", power: 30 }], evolve: "Greymon" },
    "Gabumon": { series: "Adventure", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Gabumon.png", hp: 160, atk: 38, def: 30, skills: [{ name: "Blue Blaster", power: 28 }], evolve: "Garurumon" },
    "Biyomon": { series: "Adventure", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Piyomon.png", hp: 140, atk: 42, def: 20, skills: [{ name: "Spiral Twister", power: 32 }], evolve: "Birdramon" },
    "Tentomon": { series: "Adventure", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Tentomon.png", hp: 145, atk: 35, def: 40, skills: [{ name: "Super Shocker", power: 25 }], evolve: "Kabuterimon" },
    "Palmon": { series: "Adventure", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Palmon.png", hp: 150, atk: 36, def: 35, skills: [{ name: "Poison Ivy", power: 28 }], evolve: "Togemon" },
    "Gomamon": { series: "Adventure", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Gomamon.png", hp: 180, atk: 34, def: 38, skills: [{ name: "Marching Fishes", power: 26 }], evolve: "Ikkakumon" },
    "Patamon": { series: "Adventure", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Patamon.png", hp: 130, atk: 45, def: 20, skills: [{ name: "Boom Bubble", power: 35 }], evolve: "Angemon" },
    "Gatomon": { series: "Adventure", level: "Champion", img: "https://digi-api.com/images/digimon/w/Tailmon.png", hp: 250, atk: 80, def: 40, skills: [{ name: "Lightning Paw", power: 70 }], evolve: "Angewomon" },

    "Greymon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Greymon.png", hp: 350, atk: 80, def: 60, skills: [{ name: "Nova Blast", power: 65 }], evolve: "MetalGreymon" },
    "Garurumon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Garurumon.png", hp: 330, atk: 75, def: 65, skills: [{ name: "Howling Blaster", power: 60 }], evolve: "WereGarurumon" },
    "Birdramon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Birdramon.png", hp: 310, atk: 85, def: 50, skills: [{ name: "Meteor Wing", power: 70 }], evolve: "Garudamon" },
    "Kabuterimon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Kabuterimon.png", hp: 340, atk: 70, def: 80, skills: [{ name: "Electro Shocker", power: 60 }], evolve: "MegaKabuterimon" },
    "Togemon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Togemon.png", hp: 360, atk: 72, def: 55, skills: [{ name: "Needle Spray", power: 55 }], evolve: "Lillymon" },
    "Ikkakumon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Ikkakumon.png", hp: 400, atk: 70, def: 75, skills: [{ name: "Harpoon Torpedo", power: 65 }], evolve: "Zudomon" },
    "Angemon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Angemon.png", hp: 300, atk: 100, def: 50, skills: [{ name: "Hand of Fate", power: 90 }], evolve: "MagnaAngemon" },

    "MetalGreymon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/MetalGreymon.png", hp: 700, atk: 160, def: 120, skills: [{ name: "Giga Blaster", power: 130 }], evolve: "WarGreymon" },
    "WereGarurumon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/WereGarurumon.png", hp: 650, atk: 170, def: 110, skills: [{ name: "Wolf Claw", power: 120 }], evolve: "MetalGarurumon" },
    "Garudamon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Garudamon.png", hp: 620, atk: 180, def: 100, skills: [{ name: "Wing Blade", power: 125 }], evolve: "Hououmon" },
    "MegaKabuterimon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/AtlurKabuterimon_(Red).png", hp: 680, atk: 150, def: 140, skills: [{ name: "Horn Buster", power: 115 }], evolve: "HerculesKabuterimon" },
    "Lillymon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Lilimon.png", hp: 600, atk: 165, def: 95, skills: [{ name: "Flower Cannon", power: 120 }], evolve: "Rosemon" },
    "Zudomon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Zudomon.png", hp: 750, atk: 155, def: 150, skills: [{ name: "Vulcan's Hammer", power: 130 }], evolve: "Vikemon" },
    "MagnaAngemon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/HolyAngemon.png", hp: 640, atk: 190, def: 110, skills: [{ name: "Gate of Destiny", power: 150 }], evolve: "Seraphimon" },
    "Angewomon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Angewomon.png", hp: 620, atk: 185, def: 105, skills: [{ name: "Celestial Arrow", power: 145 }], evolve: "Ophanimon" },

    "WarGreymon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/WarGreymon.png", hp: 1500, atk: 350, def: 250, skills: [{ name: "Terra Force", power: 300 }, { name: "Great Tornado", power: 220 }], evolve: null },
    "MetalGarurumon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/MetalGarurumon.png", hp: 1400, atk: 330, def: 270, skills: [{ name: "Metal Wolf Claw", power: 280 }, { name: "Giga Missile", power: 210 }], evolve: null },
    "Hououmon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Hououmon.png", hp: 1350, atk: 360, def: 230, skills: [{ name: "Starlight Explosion", power: 310 }], evolve: null },
    "HerculesKabuterimon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/HerakleKabuterimon.png", hp: 1450, atk: 320, def: 320, skills: [{ name: "Mega Electro Shocker", power: 290 }], evolve: null },
    "Rosemon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Rosemon.png", hp: 1300, atk: 340, def: 240, skills: [{ name: "Forbidden Temptation", power: 295 }], evolve: null },
    "Vikemon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Vikemon.png", hp: 1600, atk: 310, def: 350, skills: [{ name: "Arctic Blizzard", power: 285 }], evolve: null },
    "Seraphimon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Seraphimon.png", hp: 1320, atk: 380, def: 240, skills: [{ name: "Seven Heavens", power: 330 }], evolve: null },
    "Ophanimon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Ofanimon.png", hp: 1300, atk: 370, def: 250, skills: [{ name: "Eden's Javelin", power: 320 }], evolve: null },

    // --- Adventure 02 (Power Digimon) ---
    "Veemon": { series: "02", level: "Rookie", img: "https://digi-api.com/images/digimon/w/V-mon.png", hp: 160, atk: 45, def: 25, skills: [{ name: "V-Headbutt", power: 30 }], evolve: "ExVeemon" },
    "Hawkmon": { series: "02", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Hawkmon.png", hp: 140, atk: 40, def: 22, skills: [{ name: "Feather Slash", power: 28 }], evolve: "Aquilamon" },
    "Armadillomon": { series: "02", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Armadimon.png", hp: 180, atk: 35, def: 45, skills: [{ name: "Diamond Shell", power: 25 }], evolve: "Ankylomon" },
    "Wormmon": { series: "02", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Wormmon.png", hp: 130, atk: 42, def: 18, skills: [{ name: "Silk Thread", power: 26 }], evolve: "Stingmon" },

    "ExVeemon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/V-mon.png", hp: 380, atk: 85, def: 55, skills: [{ name: "V-Laser", power: 70 }], evolve: "Paildramon" },
    "Aquilamon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Aquilamon.png", hp: 320, atk: 78, def: 48, skills: [{ name: "Grand Horn", power: 65 }], evolve: "Silphymon" },
    "Ankylomon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Ankylomon.png", hp: 450, atk: 65, def: 85, skills: [{ name: "Tail Hammer", power: 60 }], evolve: "Shakkoumon" },
    "Stingmon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Stingmon.png", hp: 340, atk: 90, def: 40, skills: [{ name: "Spiking Strike", power: 75 }], evolve: "Paildramon" },

    "Paildramon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Paildramon.png", hp: 750, atk: 180, def: 130, skills: [{ name: "Desperado Blaster", power: 150 }], evolve: "Imperialdramon" },
    "Silphymon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Silphymon.png", hp: 680, atk: 175, def: 110, skills: [{ name: "Static Force", power: 140 }], evolve: "Valkyrimon" },
    "Shakkoumon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Shakkoumon.png", hp: 900, atk: 140, def: 200, skills: [{ name: "Kachina Bombs", power: 120 }], evolve: "Vikemon" },
    "Imperialdramon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Imperialdramon_Dragon_Mode.png", hp: 1600, atk: 400, def: 300, skills: [{ name: "Positron Laser", power: 350 }, { name: "Giga Death", power: 450 }], evolve: null },

    // --- Tamers ---
    "Guilmon": { series: "Tamers", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Guilmon.png", hp: 170, atk: 50, def: 30, skills: [{ name: "Fire Ball", power: 35 }], evolve: "Growlmon" },
    "Terriermon": { series: "Tamers", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Terriermon.png", hp: 140, atk: 38, def: 28, skills: [{ name: "Bunny Blast", power: 30 }], evolve: "Gargomon" },
    "Renamon": { series: "Tamers", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Renamon.png", hp: 150, atk: 55, def: 20, skills: [{ name: "Diamond Storm", power: 40 }], evolve: "Kyubimon" },
    "Impmon": { series: "Tamers", level: "Rookie", img: "https://digi-api.com/images/digimon/w/Impmon.png", hp: 120, atk: 60, def: 15, skills: [{ name: "Bada Boom", power: 45 }], evolve: "Beelzemon" },

    "Growlmon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Growmon.png", hp: 400, atk: 100, def: 60, skills: [{ name: "Pyro Blaster", power: 85 }], evolve: "WarGrowlmon" },
    "Gargomon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Galgomon.png", hp: 380, atk: 85, def: 75, skills: [{ name: "Gargo Laser", power: 80 }], evolve: "Rapidmon" },
    "Kyubimon": { level: "Champion", img: "https://digi-api.com/images/digimon/w/Kyubimon.png", hp: 350, atk: 110, def: 45, skills: [{ name: "Dragon Wheel", power: 90 }], evolve: "Taomon" },

    "WarGrowlmon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/MegaloGrowmon.png", hp: 800, atk: 200, def: 150, skills: [{ name: "Atomic Blaster", power: 180 }], evolve: "Gallantmon" },
    "Rapidmon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Rapidmon.png", hp: 750, atk: 180, def: 140, skills: [{ name: "Rapid Fire", power: 170 }], evolve: "MegaGargomon" },
    "Taomon": { level: "Ultimate", img: "https://digi-api.com/images/digimon/w/Taomon.png", hp: 700, atk: 220, def: 120, skills: [{ name: "Talisman Star", power: 190 }], evolve: "Sakuyamon" },

    "Gallantmon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Dukemon.png", hp: 1800, atk: 450, def: 350, skills: [{ name: "Shield of the Just", power: 400 }, { name: "Lightning Joust", power: 380 }], evolve: null },
    "MegaGargomon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/SaintGalgomon.png", hp: 2000, atk: 420, def: 400, skills: [{ name: "Giant Missile", power: 420 }], evolve: null },
    "Sakuyamon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Sakuyamon.png", hp: 1600, atk: 480, def: 300, skills: [{ name: "Spirit Strike", power: 440 }], evolve: null },
    "Beelzemon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/Beelzebumon.png", hp: 1700, atk: 500, def: 250, skills: [{ name: "Double Impact", power: 400 }, { name: "Corona Destroyer", power: 550 }], evolve: null },

    // --- Frontier ---
    "Agunimon": { series: "Frontier", level: "Human Spirit", img: "https://digi-api.com/images/digimon/w/Agnimon.png", hp: 200, atk: 60, def: 40, skills: [{ name: "Pyro Punch", power: 50 }], evolve: "BurningGreymon" },
    "Lobomon": { series: "Frontier", level: "Human Spirit", img: "https://digi-api.com/images/digimon/w/Wolfmon.png", hp: 190, atk: 65, def: 35, skills: [{ name: "Lobo Kendo", power: 55 }], evolve: "KendoGarurumon" },
    "Kazemon": { series: "Frontier", level: "Human Spirit", img: "https://digi-api.com/images/digimon/w/Fairimon.png", hp: 180, atk: 55, def: 30, skills: [{ name: "Hurricane Wave", power: 45 }], evolve: "Zephyrmon" },
    "Beetlemon": { series: "Frontier", level: "Human Spirit", img: "https://digi-api.com/images/digimon/w/Blitzmon.png", hp: 220, atk: 50, def: 55, skills: [{ name: "Thunder Fist", power: 40 }], evolve: "MetalKabuterimon" },
    "Kumamon": { series: "Frontier", level: "Human Spirit", img: "https://digi-api.com/images/digimon/w/Chackmon.png", hp: 210, atk: 45, def: 60, skills: [{ name: "Blizzard Blaster", power: 42 }], evolve: "Korikakumon" },

    "BurningGreymon": { level: "Beast Spirit", img: "https://digi-api.com/images/digimon/w/Vritramon.png", hp: 450, atk: 110, def: 70, skills: [{ name: "Wildfire Tsunami", power: 100 }], evolve: "EmperorGreymon" },
    "KendoGarurumon": { level: "Beast Spirit", img: "https://digi-api.com/images/digimon/w/Garmmon.png", hp: 430, atk: 120, def: 65, skills: [{ name: "Lupine Laser", power: 105 }], evolve: "MagnaGarurumon" },
    
    "EmperorGreymon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/KaiserGreymon.png", hp: 2000, atk: 550, def: 400, skills: [{ name: "Dragonfire Crossbow", power: 500 }], evolve: null },
    "MagnaGarurumon": { level: "Mega", img: "https://digi-api.com/images/digimon/w/MagnaGarurumon.png", hp: 1900, atk: 520, def: 450, skills: [{ name: "Starlight Velocity", power: 480 }], evolve: null },
};

const ENEMIES = {
    // Common Enemies
    "Kuwagamon": { hp: 180, atk: 45, def: 25, img: "https://digi-api.com/images/digimon/w/Kuwagamon.png", skills: [{name: "Scissor Claw", power: 35}], exp_yield: 40, drop: "HP Potion" },
    "Shellmon": { hp: 220, atk: 50, def: 55, img: "https://digi-api.com/images/digimon/w/Shellmon.png", skills: [{name: "Hydro Blaster", power: 40}], exp_yield: 50, drop: "HP Potion" },
    "Seadramon": { hp: 350, atk: 55, def: 50, img: "https://digi-api.com/images/digimon/w/Seadramon.png", skills: [{name: "Ice Blast", power: 45}], exp_yield: 80, drop: "Mega Potion" },
    "Gazimon": { hp: 300, atk: 70, def: 50, img: "https://digi-api.com/images/digimon/w/Gazimon.png", skills: [{name: "Electric Stun", power: 60}], exp_yield: 60, drop: "HP Potion" },
    "Flymon": { hp: 400, atk: 90, def: 40, img: "https://digi-api.com/images/digimon/w/Flymon.png", skills: [{name: "Brown Sting", power: 70}], exp_yield: 100, drop: "HP Potion" },
    "Bakemon": { hp: 380, atk: 85, def: 30, img: "https://digi-api.com/images/digimon/w/Bakemon.png", skills: [{name: "Dark Claw", power: 80}], exp_yield: 90 },

    // Villains
    "Devimon": { hp: 800, atk: 110, def: 80, img: "https://digi-api.com/images/digimon/w/Devimon.png", skills: [{name: "Death Claw", power: 100}], exp_yield: 300, isBoss: true, drop: "Evo Shard" },
    "Etemon": { hp: 1200, atk: 180, def: 120, img: "https://digi-api.com/images/digimon/w/Etemon.png", skills: [{name: "Dark Network", power: 150}], exp_yield: 500, isBoss: true, drop: "Mega Potion" },
    "Myotismon": { hp: 2000, atk: 250, def: 180, img: "https://digi-api.com/images/digimon/w/Vamdemon.png", skills: [{name: "Grisly Wing", power: 200}], exp_yield: 800, isBoss: true, drop: "Evo Shard" },
    "Apocalymon": { hp: 8000, atk: 600, def: 500, img: "https://digi-api.com/images/digimon/w/Apocalymon.png", skills: [{name: "Darkness Zone", power: 500}], exp_yield: 10000, isBoss: true },
    "BlackWarGreymon": { hp: 3000, atk: 400, def: 300, img: "https://digi-api.com/images/digimon/w/BlackWarGreymon.png", skills: [{name: "Terra Destroyer", power: 380}], exp_yield: 2000, isBoss: true },
    "MaloMyotismon": { hp: 5000, atk: 550, def: 400, img: "https://digi-api.com/images/digimon/w/BelialVamdemon.png", skills: [{name: "Crimson Mist", power: 450}], exp_yield: 4000, isBoss: true },
    "Lucemon": { hp: 10000, atk: 800, def: 600, img: "https://digi-api.com/images/digimon/w/Lucemon.png", skills: [{name: "Paradise Lost", power: 600}], exp_yield: 20000, isBoss: true },
};

// Map Grid Legend: 0: Grass, 1: Wall, 2: Water, 3: Boss, 4: Portal, 5: Road, 6: Chest
const MAPS = {
    "file_island": {
        name: "File Island",
        width: 15, height: 15,
        grid: [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,1,0,0,0,0,0,6,0,0,3,1], 
            [1,0,1,0,1,0,1,1,1,1,1,1,0,5,1],
            [1,0,1,0,0,0,0,0,0,0,0,1,0,5,1],
            [1,0,1,1,1,1,1,1,1,1,0,1,0,5,1],
            [1,0,0,0,0,6,0,0,0,1,0,1,0,5,1],
            [1,5,5,5,5,5,5,5,0,1,0,1,0,5,1],
            [1,5,1,1,1,1,1,5,0,1,0,1,0,5,1],
            [1,5,1,2,2,2,1,5,0,0,0,0,0,5,1],
            [1,5,1,2,2,2,1,5,1,1,1,1,1,5,1],
            [1,5,1,1,1,1,1,5,5,5,5,5,5,5,1],
            [1,5,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,4,1], 
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ],
        spawnX: 1, spawnY: 1,
        encounters: ["Kuwagamon", "Shellmon", "Seadramon", "Flymon", "Bakemon"],
        events: {
            "13,1": { type: "boss", enemy: "Devimon" },
            "13,12": { type: "portal", target: "folder_continent", msg: "Entering the whirlpool to Folder Continent!" },
            "5,5": { type: "chest", item: "HP Potion", msg: "Found an HP Potion!" },
            "10,1": { type: "chest", item: "Evo Shard", msg: "Found an Evo Shard!" }
        }
    },
    "folder_continent": {
        name: "Folder Continent",
        width: 15, height: 15,
        grid: [
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [2,3,0,0,0,0,0,0,0,0,0,0,0,3,2], 
            [2,0,1,1,1,1,1,1,1,1,1,1,0,1,2],
            [2,0,1,6,0,0,0,0,0,0,0,1,0,1,2],
            [2,0,1,0,1,1,1,1,1,1,0,1,0,1,2],
            [2,0,1,0,1,3,0,0,0,1,0,1,0,1,2], 
            [2,0,1,0,1,1,1,1,0,1,0,1,0,1,2],
            [2,0,1,0,0,0,0,0,0,1,0,1,0,1,2],
            [2,0,1,1,1,1,1,1,1,1,0,1,0,1,2],
            [2,0,0,0,0,0,0,0,6,0,0,1,0,1,2],
            [2,1,1,1,1,1,1,1,1,1,1,1,0,1,2],
            [2,5,5,5,5,5,5,5,5,5,5,5,5,5,2],
            [2,5,1,1,1,1,1,1,1,1,1,1,1,4,2], 
            [2,5,0,0,0,0,0,0,0,3,0,0,0,0,2],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        ],
        spawnX: 1, spawnY: 13,
        encounters: ["Gazimon"],
        events: {
            "1,1": { type: "boss", enemy: "Etemon" },
            "13,1": { type: "boss", enemy: "BlackWarGreymon" },
            "5,5": { type: "boss", enemy: "Myotismon" },
            "9,13": { type: "boss", enemy: "MaloMyotismon" },
            "13,12": { type: "portal", target: "spiral_mountain", msg: "Climbing the Spiral Mountain!" },
            "3,3": { type: "chest", item: "Mega Potion", msg: "Found a Mega Potion!" },
            "8,9": { type: "chest", item: "Evo Shard", msg: "Found an Evo Shard!" }
        }
    },
    "spiral_mountain": {
        name: "Spiral Mountain",
        width: 15, height: 15,
        grid: [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,3,5,5,5,5,5,5,5,5,5,5,5,3,1], 
            [1,5,1,1,1,1,1,1,1,1,1,1,1,5,1],
            [1,5,1,0,0,6,0,0,0,0,0,0,1,5,1],
            [1,5,1,0,1,1,1,1,1,1,1,0,1,5,1],
            [1,5,1,0,1,0,0,0,0,0,1,0,1,5,1],
            [1,5,1,0,1,0,3,1,3,0,1,0,1,5,1], 
            [1,5,1,0,1,1,1,1,1,1,1,0,1,5,1],
            [1,5,1,0,0,0,0,0,0,6,0,0,1,5,1],
            [1,5,1,1,1,1,1,1,1,1,1,1,1,5,1],
            [1,5,5,5,5,3,5,3,5,5,5,5,5,5,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,3,0,0,0,0,0,0,0,0,0,0,0,3,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ],
        spawnX: 7, spawnY: 13,
        encounters: ["Gazimon"],
        events: {
            "1,1": { type: "boss", enemy: "Apocalymon" },
            "13,1": { type: "boss", enemy: "Lucemon" },
            "5,3": { type: "chest", item: "Mega Potion", msg: "Found a Mega Potion!" },
            "9,8": { type: "chest", item: "Mega Potion", msg: "Found a Mega Potion!" }
        }
    }
};