import sys
import random
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QKeyEvent
from PyQt6.QtCore import Qt, QTimer, QRectF

from models import DataManager, MapData

TILE_SIZE = 40

class TileFactory:
    @staticmethod
    def generate_tile(tile_type: int) -> QPixmap:
        pixmap = QPixmap(TILE_SIZE, TILE_SIZE)
        painter = QPainter(pixmap)
        
        if tile_type == 0:  # Grass
            painter.fillRect(0, 0, TILE_SIZE, TILE_SIZE, QColor("#8bac0f"))
            # Add some grass details (Overlay)
            painter.setPen(QColor("#9bbc0f"))
            for i in range(0, TILE_SIZE, 10):
                for j in range(0, TILE_SIZE, 10):
                    if (i+j) % 3 == 0:
                        painter.drawLine(i, j+5, i+3, j)
        elif tile_type == 1:  # Wall / Trees
            painter.fillRect(0, 0, TILE_SIZE, TILE_SIZE, QColor("#306230"))
            painter.setPen(QColor("#0f380f"))
            painter.drawRect(2, 2, TILE_SIZE-4, TILE_SIZE-4)
            painter.drawLine(2, 2, TILE_SIZE-2, TILE_SIZE-2)
            painter.drawLine(TILE_SIZE-2, 2, 2, TILE_SIZE-2)
        elif tile_type == 2:  # Water
            painter.fillRect(0, 0, TILE_SIZE, TILE_SIZE, QColor("#0f380f"))
            painter.setPen(QColor("#306230"))
            for i in range(0, TILE_SIZE, 8):
                painter.drawArc(i, 10, 10, 10, 0, 180 * 16)
                painter.drawArc(i+4, 25, 10, 10, 0, 180 * 16)
        elif tile_type == 3:  # Boss
            painter.fillRect(0, 0, TILE_SIZE, TILE_SIZE, QColor("#c0392b"))
            painter.setPen(Qt.GlobalColor.black)
            painter.drawText(QRectF(0, 0, TILE_SIZE, TILE_SIZE), Qt.AlignmentFlag.AlignCenter, "BOSS")
        elif tile_type == 4:  # Portal
            painter.fillRect(0, 0, TILE_SIZE, TILE_SIZE, QColor("#2980b9"))
            painter.setPen(Qt.GlobalColor.white)
            painter.drawEllipse(5, 5, TILE_SIZE-10, TILE_SIZE-10)
        elif tile_type == 5:  # Road
            painter.fillRect(0, 0, TILE_SIZE, TILE_SIZE, QColor("#9bbc0f"))
            painter.setPen(QColor("#8bac0f"))
            painter.drawRect(0, 0, TILE_SIZE, TILE_SIZE)
        elif tile_type == 6:  # Chest
            painter.fillRect(0, 0, TILE_SIZE, TILE_SIZE, QColor("#f39c12"))
            painter.setPen(Qt.GlobalColor.black)
            painter.fillRect(5, 10, TILE_SIZE-10, TILE_SIZE-20, QColor("#d35400"))
            painter.drawRect(5, 10, TILE_SIZE-10, TILE_SIZE-20)
            
        painter.end()
        return pixmap

class Player(QGraphicsPixmapItem):
    def __init__(self, x: int, y: int, data_manager: DataManager):
        super().__init__()
        self.dm = data_manager
        self.grid_x = x
        self.grid_y = y
        self.setZValue(10)
        self.setPos(self.grid_x * TILE_SIZE, self.grid_y * TILE_SIZE)

    def set_sprite(self, name: str):
        path = self.dm.get_asset_path(name)
        if path:
            pix = QPixmap(path)
            pix = pix.scaled(TILE_SIZE, TILE_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(pix)
        else:
            # Fallback
            pix = QPixmap(TILE_SIZE, TILE_SIZE)
            pix.fill(Qt.GlobalColor.transparent)
            p = QPainter(pix)
            p.setBrush(QColor("#e74c3c"))
            p.drawEllipse(4, 4, TILE_SIZE-8, TILE_SIZE-8)
            p.end()
            self.setPixmap(pix)

    def move_to(self, target_x, target_y):
        self.grid_x = target_x
        self.grid_y = target_y
        self.setPos(self.grid_x * TILE_SIZE, self.grid_y * TILE_SIZE)

class MapEngine(QGraphicsView):
    def __init__(self, data_manager: DataManager, on_battle_trigger, on_portal, start_map_id: str = "file_island"):
        super().__init__()
        self.dm = data_manager
        self.on_battle_trigger = on_battle_trigger
        self.on_portal = on_portal
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedSize(600, 600)
        self.setBackgroundBrush(QColor("#000000"))
        
        self.current_map: MapData = None
        self.player = None
        self.player_digimon_name = "Agumon" # Default
        
        self.load_map(start_map_id)

    def set_player_digimon(self, name: str):
        self.player_digimon_name = name
        if self.player:
            self.player.set_sprite(name)

    def load_map(self, map_id: str):
        self.scene.clear()
        self.current_map = self.dm.maps[map_id]
        
        width = self.current_map.width
        height = self.current_map.height
        
        # Base Layer rendering
        for y in range(height):
            for x in range(width):
                tile_type = self.current_map.grid[y][x]
                pixmap = TileFactory.generate_tile(tile_type)
                item = QGraphicsPixmapItem(pixmap)
                item.setPos(x * TILE_SIZE, y * TILE_SIZE)
                self.scene.addItem(item)
                
        # Player Setup
        self.player = Player(self.current_map.spawnX, self.current_map.spawnY, self.dm)
        self.player.set_sprite(self.player_digimon_name)
        self.scene.addItem(self.player)
        
        self.centerOn(self.player)

    def keyPressEvent(self, event: QKeyEvent):
        if not self.player or not self.current_map:
            return

        dx, dy = 0, 0
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_W):
            dy = -1
        elif event.key() in (Qt.Key.Key_Down, Qt.Key.Key_S):
            dy = 1
        elif event.key() in (Qt.Key.Key_Left, Qt.Key.Key_A):
            dx = -1
        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_D):
            dx = 1

        if dx != 0 or dy != 0:
            nx = self.player.grid_x + dx
            ny = self.player.grid_y + dy
            
            if 0 <= nx < self.current_map.width and 0 <= ny < self.current_map.height:
                tile = self.current_map.grid[ny][nx]
                # 1: Wall, 2: Water (Blocking tiles)
                if tile not in (1, 2):
                    self.player.move_to(nx, ny)
                    self.centerOn(self.player)
                    self.check_event(nx, ny)

    def check_event(self, x, y):
        # Basic event trigger
        event_key = f"{x},{y}"
        if event_key in self.current_map.events:
            event = self.current_map.events[event_key]
            if event.type == "battle":
                self.on_battle_trigger(event.enemy)
            elif event.type == "portal":
                self.on_portal(event.target, event.msg)
            elif event.type == "chest":
                # For now, just show a message or logic to add item
                print(f"Chest: {event.msg} (Item: {event.item})")
        
        # Random encounter logic (e.g., 10% chance on grass)
        tile = self.current_map.grid[y][x]
        if tile == 0 and self.current_map.encounters:
            if random.random() < 0.1:
                enemy_name = random.choice(self.current_map.encounters)
                self.on_battle_trigger(enemy_name)
