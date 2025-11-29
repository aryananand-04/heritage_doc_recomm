"""
Heritage Domain Ontology

This module provides a comprehensive heritage domain ontology based on UNESCO,
ICOMOS, and ASI (Archaeological Survey of India) standards for entity linking,
disambiguation, and semantic enrichment.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class HeritageEntity:
    """Canonical heritage entity with metadata"""
    canonical_name: str
    entity_type: str  # location, person, organization, monument, event, period, style
    aliases: List[str]  # Alternative names
    description: str
    attributes: Dict[str, str]  # Additional metadata
    related_entities: List[str]  # Related canonical entities


class HeritageOntology:
    """
    Heritage domain ontology for entity linking and semantic enrichment.

    Based on:
    - UNESCO World Heritage Site categories
    - ICOMOS heritage classification
    - ASI monument categorization
    - Indian historical periods
    """

    def __init__(self):
        self.entities: Dict[str, HeritageEntity] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        self.type_hierarchy: Dict[str, List[str]] = {}

        # Initialize ontology
        self._build_heritage_types()
        self._build_architectural_styles()
        self._build_time_periods()
        self._build_regions()
        self._build_dynasties()
        self._build_religions()
        self._build_landmarks()
        self._build_persons()

    def _build_heritage_types(self):
        """Heritage type taxonomy"""
        heritage_types = {
            'monument': HeritageEntity(
                canonical_name='monument',
                entity_type='heritage_type',
                aliases=['monuments', 'memorial', 'memorials', 'structure', 'edifice'],
                description='Physical built heritage structure',
                attributes={'tangibility': 'tangible', 'category': 'built'},
                related_entities=['architecture', 'site', 'artifact']
            ),
            'site': HeritageEntity(
                canonical_name='site',
                entity_type='heritage_type',
                aliases=['sites', 'location', 'place', 'area', 'complex'],
                description='Heritage site or archaeological area',
                attributes={'tangibility': 'tangible', 'category': 'area'},
                related_entities=['monument', 'landscape']
            ),
            'architecture': HeritageEntity(
                canonical_name='architecture',
                entity_type='heritage_type',
                aliases=['architectural', 'building', 'construction'],
                description='Architectural heritage',
                attributes={'tangibility': 'tangible', 'category': 'built'},
                related_entities=['monument', 'style']
            ),
            'artifact': HeritageEntity(
                canonical_name='artifact',
                entity_type='heritage_type',
                aliases=['artifacts', 'relic', 'relics', 'object', 'item'],
                description='Movable heritage object',
                attributes={'tangibility': 'tangible', 'category': 'movable'},
                related_entities=['art', 'monument']
            ),
            'art': HeritageEntity(
                canonical_name='art',
                entity_type='heritage_type',
                aliases=['artwork', 'artistic', 'sculpture', 'painting', 'carving'],
                description='Artistic heritage',
                attributes={'tangibility': 'tangible', 'category': 'creative'},
                related_entities=['artifact', 'architecture']
            ),
            'tradition': HeritageEntity(
                canonical_name='tradition',
                entity_type='heritage_type',
                aliases=['traditions', 'custom', 'customs', 'practice', 'ritual'],
                description='Intangible cultural heritage',
                attributes={'tangibility': 'intangible', 'category': 'living'},
                related_entities=['culture', 'religion']
            ),
        }

        for name, entity in heritage_types.items():
            self._add_entity(entity)

    def _build_architectural_styles(self):
        """Architectural style taxonomy"""
        styles = {
            'indo-islamic': HeritageEntity(
                canonical_name='indo-islamic',
                entity_type='architectural_style',
                aliases=['indo islamic', 'indo saracenic', 'muslim architecture'],
                description='Fusion of Indian and Islamic architectural elements',
                attributes={'period': 'medieval', 'region': 'india'},
                related_entities=['mughal', 'sultanate']
            ),
            'mughal': HeritageEntity(
                canonical_name='mughal',
                entity_type='architectural_style',
                aliases=['mughal architecture', 'moghul'],
                description='Mughal Empire architectural style',
                attributes={'period': 'medieval', 'region': 'india', 'dynasty': 'mughal'},
                related_entities=['indo-islamic', 'persian']
            ),
            'dravidian': HeritageEntity(
                canonical_name='dravidian',
                entity_type='architectural_style',
                aliases=['dravidian architecture', 'south indian style'],
                description='South Indian temple architecture',
                attributes={'period': 'ancient-medieval', 'region': 'south'},
                related_entities=['hindu', 'temple']
            ),
            'nagara': HeritageEntity(
                canonical_name='nagara',
                entity_type='architectural_style',
                aliases=['nagara architecture', 'north indian style'],
                description='North Indian temple architecture',
                attributes={'period': 'ancient-medieval', 'region': 'north'},
                related_entities=['hindu', 'temple']
            ),
            'vesara': HeritageEntity(
                canonical_name='vesara',
                entity_type='architectural_style',
                aliases=['vesara architecture', 'hybrid style'],
                description='Mixed Dravidian-Nagara style',
                attributes={'period': 'medieval', 'region': 'central'},
                related_entities=['dravidian', 'nagara']
            ),
            'buddhist': HeritageEntity(
                canonical_name='buddhist',
                entity_type='architectural_style',
                aliases=['buddhist architecture', 'stupa', 'vihara', 'chaitya'],
                description='Buddhist architectural tradition',
                attributes={'period': 'ancient', 'religion': 'buddhism'},
                related_entities=['rock-cut', 'monastery']
            ),
            'colonial': HeritageEntity(
                canonical_name='colonial',
                entity_type='architectural_style',
                aliases=['colonial architecture', 'british architecture', 'european style'],
                description='Colonial period architecture',
                attributes={'period': 'modern', 'region': 'india'},
                related_entities=['indo-saracenic', 'victorian']
            ),
            'rock-cut': HeritageEntity(
                canonical_name='rock-cut',
                entity_type='architectural_style',
                aliases=['rock cut', 'cave architecture', 'cave temple'],
                description='Architecture carved from rock',
                attributes={'technique': 'carving'},
                related_entities=['buddhist', 'hindu', 'jain']
            ),
        }

        for name, entity in styles.items():
            self._add_entity(entity)

    def _build_time_periods(self):
        """Historical period taxonomy"""
        periods = {
            'ancient': HeritageEntity(
                canonical_name='ancient',
                entity_type='time_period',
                aliases=['prehistoric', 'early historic', 'classical'],
                description='Ancient period (pre-600 CE)',
                attributes={'start': '-3000', 'end': '600'},
                related_entities=['mauryan', 'gupta', 'indus-valley']
            ),
            'medieval': HeritageEntity(
                canonical_name='medieval',
                entity_type='time_period',
                aliases=['middle ages', 'sultanate period', 'mughal period'],
                description='Medieval period (600-1800 CE)',
                attributes={'start': '600', 'end': '1800'},
                related_entities=['mughal', 'delhi-sultanate', 'vijayanagara']
            ),
            'modern': HeritageEntity(
                canonical_name='modern',
                entity_type='time_period',
                aliases=['colonial period', 'british period', 'contemporary'],
                description='Modern period (1800 CE onwards)',
                attributes={'start': '1800', 'end': '2025'},
                related_entities=['colonial', 'independence']
            ),
        }

        for name, entity in periods.items():
            self._add_entity(entity)

    def _build_regions(self):
        """Geographic region taxonomy"""
        regions = {
            'north india': HeritageEntity(
                canonical_name='north india',
                entity_type='region',
                aliases=['northern india', 'north', 'indo-gangetic plain'],
                description='Northern India region',
                attributes={'states': 'punjab,haryana,delhi,up,uttarakhand,hp,j&k'},
                related_entities=['mughal', 'delhi-sultanate']
            ),
            'south india': HeritageEntity(
                canonical_name='south india',
                entity_type='region',
                aliases=['southern india', 'south', 'deccan'],
                description='Southern India region',
                attributes={'states': 'tamil-nadu,karnataka,kerala,andhra-pradesh,telangana'},
                related_entities=['dravidian', 'chola', 'vijayanagara']
            ),
            'east india': HeritageEntity(
                canonical_name='east india',
                entity_type='region',
                aliases=['eastern india', 'east'],
                description='Eastern India region',
                attributes={'states': 'west-bengal,odisha,bihar,jharkhand'},
                related_entities=['pala', 'bengal-sultanate']
            ),
            'west india': HeritageEntity(
                canonical_name='west india',
                entity_type='region',
                aliases=['western india', 'west'],
                description='Western India region',
                attributes={'states': 'gujarat,maharashtra,rajasthan,goa'},
                related_entities=['maratha', 'rajput']
            ),
            'central india': HeritageEntity(
                canonical_name='central india',
                entity_type='region',
                aliases=['central', 'madhya pradesh'],
                description='Central India region',
                attributes={'states': 'madhya-pradesh,chhattisgarh'},
                related_entities=['chandela', 'gond']
            ),
        }

        for name, entity in regions.items():
            self._add_entity(entity)

    def _build_dynasties(self):
        """Dynasty and empire taxonomy"""
        dynasties = {
            'mughal empire': HeritageEntity(
                canonical_name='mughal empire',
                entity_type='dynasty',
                aliases=['mughal', 'moghul empire', 'timurid'],
                description='Mughal Dynasty (1526-1857)',
                attributes={'period': 'medieval', 'religion': 'islam', 'region': 'india'},
                related_entities=['akbar', 'shah-jahan', 'taj-mahal']
            ),
            'mauryan empire': HeritageEntity(
                canonical_name='mauryan empire',
                entity_type='dynasty',
                aliases=['maurya', 'mauryan'],
                description='Mauryan Dynasty (322-185 BCE)',
                attributes={'period': 'ancient', 'religion': 'buddhism', 'region': 'india'},
                related_entities=['ashoka', 'sanchi', 'pataliputra']
            ),
            'gupta empire': HeritageEntity(
                canonical_name='gupta empire',
                entity_type='dynasty',
                aliases=['gupta', 'guptas'],
                description='Gupta Dynasty (320-550 CE)',
                attributes={'period': 'ancient', 'region': 'north'},
                related_entities=['chandragupta', 'nalanda']
            ),
            'chola dynasty': HeritageEntity(
                canonical_name='chola dynasty',
                entity_type='dynasty',
                aliases=['chola', 'cholas'],
                description='Chola Dynasty (300 BCE-1279 CE)',
                attributes={'period': 'ancient-medieval', 'region': 'south'},
                related_entities=['brihadeeswara', 'thanjavur', 'dravidian']
            ),
            'vijayanagara empire': HeritageEntity(
                canonical_name='vijayanagara empire',
                entity_type='dynasty',
                aliases=['vijayanagar', 'vijayanagara'],
                description='Vijayanagara Empire (1336-1646)',
                attributes={'period': 'medieval', 'region': 'south'},
                related_entities=['hampi', 'krishnadevaraya']
            ),
            'delhi sultanate': HeritageEntity(
                canonical_name='delhi sultanate',
                entity_type='dynasty',
                aliases=['sultanate', 'delhi sultans'],
                description='Delhi Sultanate (1206-1526)',
                attributes={'period': 'medieval', 'religion': 'islam', 'region': 'north'},
                related_entities=['qutub-minar', 'alauddin-khilji']
            ),
            'maratha empire': HeritageEntity(
                canonical_name='maratha empire',
                entity_type='dynasty',
                aliases=['maratha', 'marathas'],
                description='Maratha Empire (1674-1818)',
                attributes={'period': 'medieval', 'region': 'west'},
                related_entities=['shivaji', 'raigad']
            ),
        }

        for name, entity in dynasties.items():
            self._add_entity(entity)

    def _build_religions(self):
        """Religious tradition taxonomy"""
        religions = {
            'hinduism': HeritageEntity(
                canonical_name='hinduism',
                entity_type='religion',
                aliases=['hindu', 'sanatana dharma'],
                description='Hindu religious tradition',
                attributes={},
                related_entities=['temple', 'dravidian', 'nagara']
            ),
            'buddhism': HeritageEntity(
                canonical_name='buddhism',
                entity_type='religion',
                aliases=['buddhist', 'buddha dharma'],
                description='Buddhist religious tradition',
                attributes={},
                related_entities=['stupa', 'vihara', 'monastery']
            ),
            'jainism': HeritageEntity(
                canonical_name='jainism',
                entity_type='religion',
                aliases=['jain'],
                description='Jain religious tradition',
                attributes={},
                related_entities=['temple', 'tirthankara']
            ),
            'islam': HeritageEntity(
                canonical_name='islam',
                entity_type='religion',
                aliases=['islamic', 'muslim'],
                description='Islamic religious tradition',
                attributes={},
                related_entities=['mosque', 'dargah', 'tomb']
            ),
            'sikhism': HeritageEntity(
                canonical_name='sikhism',
                entity_type='religion',
                aliases=['sikh'],
                description='Sikh religious tradition',
                attributes={},
                related_entities=['gurdwara', 'golden-temple']
            ),
        }

        for name, entity in religions.items():
            self._add_entity(entity)

    def _build_landmarks(self):
        """Major heritage landmarks for disambiguation"""
        landmarks = {
            'taj mahal': HeritageEntity(
                canonical_name='taj mahal',
                entity_type='monument',
                aliases=['tajmahal', 'taj'],
                description='Mughal mausoleum in Agra',
                attributes={'location': 'agra', 'state': 'uttar pradesh', 'period': 'medieval',
                           'style': 'mughal', 'type': 'tomb', 'unesco': 'yes'},
                related_entities=['shah-jahan', 'agra', 'mughal empire']
            ),
            'qutub minar': HeritageEntity(
                canonical_name='qutub minar',
                entity_type='monument',
                aliases=['qutb minar', 'qutab minar'],
                description='Minaret in Delhi',
                attributes={'location': 'delhi', 'period': 'medieval',
                           'style': 'indo-islamic', 'type': 'tower', 'unesco': 'yes'},
                related_entities=['delhi sultanate', 'delhi']
            ),
            'red fort': HeritageEntity(
                canonical_name='red fort',
                entity_type='monument',
                aliases=['lal qila', 'red fort delhi'],
                description='Mughal fort in Delhi',
                attributes={'location': 'delhi', 'period': 'medieval',
                           'style': 'mughal', 'type': 'fort', 'unesco': 'yes'},
                related_entities=['shah-jahan', 'delhi', 'mughal empire']
            ),
            'golden temple': HeritageEntity(
                canonical_name='golden temple',
                entity_type='monument',
                aliases=['harmandir sahib', 'darbar sahib'],
                description='Sikh gurdwara in Amritsar',
                attributes={'location': 'amritsar', 'state': 'punjab', 'period': 'medieval',
                           'religion': 'sikhism', 'type': 'gurdwara'},
                related_entities=['sikhism', 'amritsar']
            ),
            'ajanta caves': HeritageEntity(
                canonical_name='ajanta caves',
                entity_type='site',
                aliases=['ajanta', 'ajanta ellora'],
                description='Buddhist rock-cut caves',
                attributes={'location': 'maharashtra', 'period': 'ancient',
                           'style': 'rock-cut', 'religion': 'buddhism', 'unesco': 'yes'},
                related_entities=['buddhism', 'ellora caves']
            ),
            'ellora caves': HeritageEntity(
                canonical_name='ellora caves',
                entity_type='site',
                aliases=['ellora'],
                description='Rock-cut caves (Buddhist, Hindu, Jain)',
                attributes={'location': 'maharashtra', 'period': 'ancient-medieval',
                           'style': 'rock-cut', 'unesco': 'yes'},
                related_entities=['buddhism', 'hinduism', 'jainism', 'ajanta caves']
            ),
            'sanchi stupa': HeritageEntity(
                canonical_name='sanchi stupa',
                entity_type='monument',
                aliases=['sanchi', 'great stupa'],
                description='Buddhist stupa complex',
                attributes={'location': 'madhya pradesh', 'period': 'ancient',
                           'religion': 'buddhism', 'type': 'stupa', 'unesco': 'yes'},
                related_entities=['ashoka', 'mauryan empire', 'buddhism']
            ),
            'hampi': HeritageEntity(
                canonical_name='hampi',
                entity_type='site',
                aliases=['vijayanagara', 'hampi ruins'],
                description='Vijayanagara Empire capital ruins',
                attributes={'location': 'karnataka', 'state': 'karnataka', 'period': 'medieval',
                           'unesco': 'yes'},
                related_entities=['vijayanagara empire', 'krishnadevaraya']
            ),
            'khajuraho': HeritageEntity(
                canonical_name='khajuraho',
                entity_type='site',
                aliases=['khajuraho temples'],
                description='Temple complex with erotic sculptures',
                attributes={'location': 'madhya pradesh', 'period': 'medieval',
                           'style': 'nagara', 'religion': 'hinduism', 'unesco': 'yes'},
                related_entities=['chandela dynasty', 'nagara']
            ),
        }

        for name, entity in landmarks.items():
            self._add_entity(entity)

    def _build_persons(self):
        """Historical figures"""
        persons = {
            'ashoka': HeritageEntity(
                canonical_name='ashoka',
                entity_type='person',
                aliases=['ashoka the great', 'emperor ashoka'],
                description='Mauryan Emperor (304-232 BCE)',
                attributes={'dynasty': 'mauryan', 'period': 'ancient', 'religion': 'buddhism'},
                related_entities=['mauryan empire', 'sanchi stupa']
            ),
            'shah jahan': HeritageEntity(
                canonical_name='shah jahan',
                entity_type='person',
                aliases=['shahjahan', 'shah jehan'],
                description='Mughal Emperor (1592-1666)',
                attributes={'dynasty': 'mughal', 'period': 'medieval'},
                related_entities=['taj mahal', 'red fort', 'mughal empire']
            ),
            'akbar': HeritageEntity(
                canonical_name='akbar',
                entity_type='person',
                aliases=['akbar the great', 'emperor akbar'],
                description='Mughal Emperor (1542-1605)',
                attributes={'dynasty': 'mughal', 'period': 'medieval'},
                related_entities=['fatehpur sikri', 'mughal empire']
            ),
            'gautama buddha': HeritageEntity(
                canonical_name='gautama buddha',
                entity_type='person',
                aliases=['buddha', 'siddhartha', 'the buddha'],
                description='Founder of Buddhism',
                attributes={'religion': 'buddhism', 'period': 'ancient'},
                related_entities=['buddhism', 'bodh gaya']
            ),
        }

        for name, entity in persons.items():
            self._add_entity(entity)

    def _add_entity(self, entity: HeritageEntity):
        """Add entity to ontology with aliases"""
        self.entities[entity.canonical_name] = entity

        # Map aliases to canonical
        self.alias_to_canonical[entity.canonical_name.lower()] = entity.canonical_name
        for alias in entity.aliases:
            self.alias_to_canonical[alias.lower()] = entity.canonical_name

    def link_entity(self, mention: str, context: Optional[str] = None) -> Optional[str]:
        """
        Link entity mention to canonical entity.

        Args:
            mention: Entity mention from text
            context: Optional context for disambiguation

        Returns:
            Canonical entity name or None
        """
        mention_lower = mention.lower().strip()

        # Direct match
        if mention_lower in self.alias_to_canonical:
            return self.alias_to_canonical[mention_lower]

        # Partial match (for compound names)
        for alias, canonical in self.alias_to_canonical.items():
            if mention_lower in alias or alias in mention_lower:
                # Context-based disambiguation if needed
                if context:
                    # Simple heuristic: if context mentions related entities, prefer that match
                    entity = self.entities[canonical]
                    for related in entity.related_entities:
                        if related.lower() in context.lower():
                            return canonical
                return canonical

        return None

    def get_entity(self, canonical_name: str) -> Optional[HeritageEntity]:
        """Get entity by canonical name"""
        return self.entities.get(canonical_name)

    def get_related_entities(self, canonical_name: str) -> List[str]:
        """Get related entities for given entity"""
        entity = self.get_entity(canonical_name)
        return entity.related_entities if entity else []

    def get_entities_by_type(self, entity_type: str) -> List[HeritageEntity]:
        """Get all entities of given type"""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def normalize_entity(self, mention: str) -> str:
        """Normalize entity mention (case, whitespace)"""
        # Link if possible, otherwise normalize
        canonical = self.link_entity(mention)
        if canonical:
            return canonical

        # Fallback: basic normalization
        normalized = mention.lower().strip()
        normalized = ' '.join(normalized.split())  # Collapse whitespace
        return normalized

    def compute_semantic_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute semantic similarity between two entities.

        Returns:
            Similarity score [0, 1]
        """
        # Get canonical forms
        canonical1 = self.link_entity(entity1) or entity1
        canonical2 = self.link_entity(entity2) or entity2

        # Same entity
        if canonical1 == canonical2:
            return 1.0

        # Get entities
        e1 = self.get_entity(canonical1)
        e2 = self.get_entity(canonical2)

        if not e1 or not e2:
            return 0.0

        # Same type bonus
        similarity = 0.0
        if e1.entity_type == e2.entity_type:
            similarity += 0.3

        # Related entities bonus
        if canonical2 in e1.related_entities or canonical1 in e2.related_entities:
            similarity += 0.6

        # Shared attributes
        shared_attrs = set(e1.attributes.keys()) & set(e2.attributes.keys())
        if shared_attrs:
            matching = sum(1 for k in shared_attrs if e1.attributes[k] == e2.attributes[k])
            similarity += 0.1 * (matching / len(shared_attrs))

        return min(similarity, 1.0)

    def save_to_json(self, filepath: str):
        """Save ontology to JSON file"""
        data = {
            'entities': {
                name: {
                    'canonical_name': e.canonical_name,
                    'entity_type': e.entity_type,
                    'aliases': e.aliases,
                    'description': e.description,
                    'attributes': e.attributes,
                    'related_entities': e.related_entities
                }
                for name, e in self.entities.items()
            },
            'alias_map': self.alias_to_canonical
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'HeritageOntology':
        """Load ontology from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        ontology = cls.__new__(cls)
        ontology.entities = {}
        ontology.alias_to_canonical = data['alias_map']
        ontology.type_hierarchy = {}

        for name, e_data in data['entities'].items():
            entity = HeritageEntity(
                canonical_name=e_data['canonical_name'],
                entity_type=e_data['entity_type'],
                aliases=e_data['aliases'],
                description=e_data['description'],
                attributes=e_data['attributes'],
                related_entities=e_data['related_entities']
            )
            ontology.entities[name] = entity

        return ontology


def create_default_ontology() -> HeritageOntology:
    """Create and return default heritage ontology"""
    return HeritageOntology()


if __name__ == "__main__":
    # Create and save default ontology
    ontology = create_default_ontology()
    ontology.save_to_json("data/ontology/heritage_ontology.json")

    print(f"✓ Created heritage ontology")
    print(f"  Entities: {len(ontology.entities)}")
    print(f"  Aliases: {len(ontology.alias_to_canonical)}")
    print(f"\n  By type:")

    from collections import Counter
    type_counts = Counter(e.entity_type for e in ontology.entities.values())
    for entity_type, count in type_counts.most_common():
        print(f"    {entity_type}: {count}")
