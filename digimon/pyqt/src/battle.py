import random
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap
from PyQt6.QtCore import Qt, QPropertyAnimation, QPoint, QEasingCurve, QTimer, QSequentialAnimationGroup

from models import DataManager, DigimonData, EnemyData, Skill

class FlashySprite(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.base_pos = QPoint(0, 0)
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
    
    def set_base_pos(self, pos: QPoint):
        self.base_pos = pos
        self.move(self.base_pos)

    def shake(self):
        # Create a shaking animation
        self.shake_anim = QSequentialAnimationGroup(self)
        offsets = [(10, 0), (-10, 0), (5, 0), (-5, 0), (0, 0)]
        
        for dx, dy in offsets:
            anim = QPropertyAnimation(self, b"pos")
            anim.setDuration(50)
            anim.setEndValue(self.base_pos + QPoint(dx, dy))
            self.shake_anim.addAnimation(anim)
            
        self.shake_anim.start()

    def flash(self):
        # Flash by dropping opacity and restoring it
        self.flash_anim = QSequentialAnimationGroup(self)
        anim1 = QPropertyAnimation(self.opacity_effect, b"opacity")
        anim1.setDuration(100)
        anim1.setEndValue(0.2)
        
        anim2 = QPropertyAnimation(self.opacity_effect, b"opacity")
        anim2.setDuration(100)
        anim2.setEndValue(1.0)
        
        self.flash_anim.addAnimation(anim1)
        self.flash_anim.addAnimation(anim2)
        self.flash_anim.start()

class FloatingText(QLabel):
    def __init__(self, text, parent, color="white"):
        super().__init__(text, parent)
        self.setFont(QFont("Impact", 24, QFont.Weight.Bold))
        self.setStyleSheet(f"color: {color};")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def start_animation(self, start_pos: QPoint):
        self.move(start_pos)
        self.show()
        
        # Move up animation
        self.pos_anim = QPropertyAnimation(self, b"pos")
        self.pos_anim.setDuration(1000)
        self.pos_anim.setStartValue(start_pos)
        self.pos_anim.setEndValue(start_pos - QPoint(0, 50))
        self.pos_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        # Fade out animation
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        
        self.pos_anim.finished.connect(self.deleteLater)
        
        self.pos_anim.start()
        self.fade_anim.start()


class BattleEngine(QWidget):
    def __init__(self, data_manager: DataManager, on_battle_end):
        super().__init__()
        self.dm = data_manager
        self.on_battle_end = on_battle_end
        self.setStyleSheet("background-color: #1a1a1a; color: white;")
        
        self.player_digimon: DigimonData = None
        self.enemy: EnemyData = None
        self.player_hp = 0
        self.enemy_hp = 0
        
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Field Layout (Sprites)
        field_layout = QHBoxLayout()
        
        # Player Side
        player_side = QVBoxLayout()
        self.player_name_label = QLabel("Player", self)
        self.player_name_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.player_hp_bar = QProgressBar(self)
        self.player_hp_bar.setStyleSheet("QProgressBar::chunk { background-color: #27ae60; }")
        
        self.player_sprite = FlashySprite(self)
        player_side.addWidget(self.player_name_label)
        player_side.addWidget(self.player_hp_bar)
        player_side.addWidget(self.player_sprite, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Enemy Side
        enemy_side = QVBoxLayout()
        self.enemy_name_label = QLabel("Enemy", self)
        self.enemy_name_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.enemy_hp_bar = QProgressBar(self)
        self.enemy_hp_bar.setStyleSheet("QProgressBar::chunk { background-color: #c0392b; }")
        
        self.enemy_sprite = FlashySprite(self)
        enemy_side.addWidget(self.enemy_name_label)
        enemy_side.addWidget(self.enemy_hp_bar)
        enemy_side.addWidget(self.enemy_sprite, alignment=Qt.AlignmentFlag.AlignCenter)
        
        field_layout.addLayout(player_side)
        field_layout.addStretch()
        field_layout.addLayout(enemy_side)
        
        # Actions Layout
        self.actions_layout = QHBoxLayout()
        
        main_layout.addLayout(field_layout)
        main_layout.addLayout(self.actions_layout)

    def load_pixmap(self, name: str, size: int = 200) -> QPixmap:
        path = self.dm.get_asset_path(name)
        if path:
            pix = QPixmap(path)
            return pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        # Fallback
        pix = QPixmap(size, size)
        pix.fill(QColor("#333333"))
        return pix

    def init_battle(self, player_digimon: DigimonData, enemy_name: str):
        self.player_digimon = player_digimon
        self.enemy = self.dm.enemies[enemy_name]
        
        self.player_hp = self.player_digimon.hp
        self.enemy_hp = self.enemy.hp
        
        self.player_name_label.setText(f"{self.player_digimon.name} (Lv.{self.player_digimon.level})")
        self.enemy_name_label.setText(self.enemy.name)
        
        self.player_hp_bar.setMaximum(self.player_hp)
        self.player_hp_bar.setValue(self.player_hp)
        self.enemy_hp_bar.setMaximum(self.enemy_hp)
        self.enemy_hp_bar.setValue(self.enemy_hp)
        
        # Set Pixmaps
        self.player_sprite.setPixmap(self.load_pixmap(self.player_digimon.name))
        self.enemy_sprite.setPixmap(self.load_pixmap(self.enemy.name))
        
        # Update button setup later when base pos is set
        QTimer.singleShot(100, self.post_init_setup)
        
        # Setup skills
        for i in reversed(range(self.actions_layout.count())): 
            item = self.actions_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
            
        for skill in self.player_digimon.skills:
            btn = QPushButton(f"{skill.name}\n(Pow: {skill.power})", self)
            btn.setMinimumHeight(60)
            btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            btn.setStyleSheet("background-color: #2c3e50; border-radius: 10px; padding: 10px;")
            btn.clicked.connect(lambda checked, s=skill: self.player_attack(s))
            self.actions_layout.addWidget(btn)

    def post_init_setup(self):
        self.player_sprite.set_base_pos(self.player_sprite.pos())
        self.enemy_sprite.set_base_pos(self.enemy_sprite.pos())

    def get_type_multiplier(self, attacker_type, defender_type):
        # Vaccine > Virus > Data > Vaccine
        advantages = {
            "Vaccine": "Virus",
            "Virus": "Data",
            "Data": "Vaccine"
        }
        if advantages.get(attacker_type) == defender_type:
            return 1.5
        if advantages.get(defender_type) == attacker_type:
            return 0.5
        return 1.0

    def player_attack(self, skill: Skill):
        # Type Multiplier
        mult = self.get_type_multiplier(self.player_digimon.type, self.enemy.type)
        
        # Damage calculation
        base_dmg = max(10, int(self.player_digimon.atk * (skill.power / 50) - self.enemy.def_stat / 2))
        dmg = int(base_dmg * mult * random.uniform(0.9, 1.1))
        
        self.enemy_hp = max(0, self.enemy_hp - dmg)
        self.enemy_hp_bar.setValue(self.enemy_hp)
        
        # Effects!
        self.enemy_sprite.shake()
        self.enemy_sprite.flash()
        
        # Floating Text (Critical styling if advantage)
        color = "red" if mult > 1.0 else ("gray" if mult < 1.0 else "yellow")
        msg = f"{dmg}!" if mult == 1.0 else (f"{dmg} CRITICAL!" if mult > 1.0 else f"{dmg} weak")
        float_txt = FloatingText(msg, self, color)
        
        target_pos = self.enemy_sprite.pos() + QPoint(75, 75)
        float_txt.start_animation(target_pos)
        
        if self.enemy_hp <= 0:
            QTimer.singleShot(1500, self.win_battle)
        else:
            self.set_buttons_enabled(False)
            QTimer.singleShot(1500, self.enemy_attack)

    def enemy_attack(self):
        skill = random.choice(self.enemy.skills)
        mult = self.get_type_multiplier(self.enemy.type, self.player_digimon.type)
        
        base_dmg = max(5, int(self.enemy.atk * (skill.power / 50) - self.player_digimon.def_stat / 2))
        dmg = int(base_dmg * mult * random.uniform(0.9, 1.1))
        
        self.player_hp = max(0, self.player_hp - dmg)
        self.player_hp_bar.setValue(self.player_hp)
        
        self.player_sprite.shake()
        self.player_sprite.flash()
        
        color = "red" if mult > 1.0 else "white"
        float_txt = FloatingText(str(dmg), self, color)
        target_pos = self.player_sprite.pos() + QPoint(75, 75)
        float_txt.start_animation(target_pos)
        
        if self.player_hp <= 0:
            QTimer.singleShot(1500, self.lose_battle)
        else:
            self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled):
        for i in range(self.actions_layout.count()):
            widget = self.actions_layout.itemAt(i).widget()
            if widget:
                widget.setEnabled(enabled)

    def win_battle(self):
        self.on_battle_end(True)
        
    def lose_battle(self):
        self.on_battle_end(False)
