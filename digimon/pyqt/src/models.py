import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

@dataclass
class Skill:
    name: str
    power: int

@dataclass
class DigimonData:
    name: str
    level: str
    img: str
    hp: int
    atk: int
    def_stat: int
    type: str # Vaccine, Data, Virus
    skills: List[Skill]
    evolve: Optional[str] = None
    series: Optional[str] = None

@dataclass
class EnemyData:
    name: str
    hp: int
    atk: int
    def_stat: int
    type: str # Vaccine, Data, Virus
    img: str
    skills: List[Skill]
    exp_yield: int
    drop: Optional[str] = None
    isBoss: bool = False

@dataclass
class ItemData:
    name: str
    type: str
    value: int
    desc: str

@dataclass
class MapEvent:
    type: str
    target: Optional[str] = None
    msg: Optional[str] = None
    enemy: Optional[str] = None
    item: Optional[str] = None

@dataclass
class MapData:
    id: str
    name: str
    width: int
    height: int
    grid: List[List[int]]
    spawnX: int
    spawnY: int
    encounters: List[str]
    events: Dict[str, MapEvent]

class DataManager:
    def __init__(self):
        # Determine base path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, 'data')
        self.assets_dir = os.path.join(base_dir, 'assets')
        self.digimons: Dict[str, DigimonData] = {}
        self.enemies: Dict[str, EnemyData] = {}
        self.items: Dict[str, ItemData] = {}
        self.maps: Dict[str, MapData] = {}
        self.load_all()

    def get_asset_path(self, name: str) -> str:
        path = os.path.join(self.assets_dir, f"{name}.png")
        if os.path.exists(path):
            return path
        return ""

    def load_all(self):
        # Load Digimons
        with open(f"{self.data_dir}/digimon.json", 'r', encoding='utf-8') as f:
            d_data = json.load(f)
            for k, v in d_data.items():
                skills = [Skill(**s) for s in v.get('skills', [])]
                self.digimons[k] = DigimonData(
                    name=k,
                    level=v['level'],
                    img=v['img'],
                    hp=v['hp'],
                    atk=v['atk'],
                    def_stat=v['def'],
                    type=v.get('type', 'Data'),
                    skills=skills,
                    evolve=v.get('evolve'),
                    series=v.get('series')
                )

        # Load Enemies
        with open(f"{self.data_dir}/enemies.json", 'r', encoding='utf-8') as f:
            e_data = json.load(f)
            for k, v in e_data.items():
                skills = [Skill(**s) for s in v.get('skills', [])]
                self.enemies[k] = EnemyData(
                    name=k,
                    hp=v['hp'],
                    atk=v['atk'],
                    def_stat=v['def'],
                    type=v.get('type', 'Virus'),
                    img=v['img'],
                    skills=skills,
                    exp_yield=v['exp_yield'],
                    drop=v.get('drop'),
                    isBoss=v.get('isBoss', False)
                )

        # Load Items
        with open(f"{self.data_dir}/items.json", 'r', encoding='utf-8') as f:
            i_data = json.load(f)
            for k, v in i_data.items():
                self.items[k] = ItemData(name=k, **v)

        # Load Maps
        with open(f"{self.data_dir}/maps.json", 'r', encoding='utf-8') as f:
            m_data = json.load(f)
            for k, v in m_data.items():
                events = {ek: MapEvent(**ev) for ek, ev in v.get('events', {}).items()}
                self.maps[k] = MapData(
                    id=k,
                    name=v['name'],
                    width=v['width'],
                    height=v['height'],
                    grid=v['grid'],
                    spawnX=v['spawnX'],
                    spawnY=v['spawnY'],
                    encounters=v.get('encounters', []),
                    events=events
                )
