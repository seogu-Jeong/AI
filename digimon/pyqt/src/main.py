import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from models import DataManager
from engine import MapEngine
from battle import BattleEngine
from ui.title import TitleScreen

class DigimonGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digimon RPG - PyQt6 Upgrade")
        self.setFixedSize(600, 600)
        
        # Data Manager
        self.dm = DataManager()
        
        # Central Stacked Widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # Initialize Screens
        self.title_screen = TitleScreen(on_start=self.start_game)
        self.map_engine = MapEngine(self.dm, on_battle_trigger=self.enter_battle, on_portal=self.change_map, start_map_id="file_island")
        self.battle_engine = BattleEngine(self.dm, on_battle_end=self.exit_battle)
        
        # Add to stack
        self.stack.addWidget(self.title_screen) # Index 0
        self.stack.addWidget(self.map_engine)   # Index 1
        self.stack.addWidget(self.battle_engine) # Index 2
        
        # Start directly at Map for testing
        self.stack.setCurrentIndex(1)
        self.map_engine.setFocus()

    def start_game(self):
        self.stack.setCurrentIndex(1)
        self.map_engine.setFocus()

    def change_map(self, target_map_id, msg):
        QMessageBox.information(self, "Portal", msg)
        self.map_engine.load_map(target_map_id)
        self.map_engine.setFocus()

    def enter_battle(self, enemy_name):
        # For now, use Agumon as player's digimon
        player_digimon = self.dm.digimons["Agumon"]
        self.battle_engine.init_battle(player_digimon, enemy_name)
        self.stack.setCurrentIndex(2)

    def exit_battle(self, won):
        print(f"Battle ended. Won: {won}")
        self.stack.setCurrentIndex(1)
        self.map_engine.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigimonGame()
    window.show()
    sys.exit(app.exec())
