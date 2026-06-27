from random import sample

from numpy import clip
from numpy.random import default_rng, Generator

from vgc2.balance.meta import MoveSet
from vgc2.battle_engine import Type
from vgc2.battle_engine.modifiers import Category, Nature
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies, Pokemon
from vgc2.battle_engine.team import Team
from vgc2.util.generator import MoveGenerator, MoveSetGenerator, PokemonSpeciesGenerator, PokemonGenerator, \
    gen_pkm_species

_RNG = default_rng()


def gen_move(rng: Generator = _RNG) -> Move:
    category = Category(rng.choice(len(Category) - 1, 1, False) + 1)
    base_power = 0 if category == Category.OTHER else int(clip(rng.normal(100, 40, 1)[0], 0, 140))
    return Move(
        pkm_type=Type(rng.choice(len(Type) - 1, 1, False)),  # no typeless
        base_power=base_power,
        accuracy=1. if rng.random() < .5 else float(rng.uniform(.75, 1.)),
        max_pp=int(clip(rng.normal(10, 2, 1)[0], 5, 20)),
        category=category)


def gen_move_set(n: int,
                 rng: Generator = _RNG,
                 _gen_move: MoveGenerator = gen_move) -> MoveSet:
    return [_gen_move(rng) for _ in range(n)]


"""def gen_pkm(species: PokemonSpecies,
            max_moves: int = 4,
            rng: Generator = _RNG) -> Pokemon:
    for i, t in enumerate(species.types):
        if i < len(species.moves):
            # Overwriting the move type to match the species type
            species.moves[i].pkm_type = t
    n_moves = len(species.moves)
    return Pokemon(
        species=species,
        move_indexes=list(sample([i for i in range(n_moves)], min(max_moves, n_moves))),
        level=100,
        ivs=(31,) * 6,
        evs=tuple(list(int(x) for x in rng.multinomial(510, [1 / 6] * 6))),
        nature=Nature(rng.choice(len(Nature), 1)[0]))"""


def gen_pkm(species: PokemonSpecies,
            max_moves: int = 4,
            rng: Generator = _RNG) -> Pokemon:
    # Force STAB moves
    for i, t in enumerate(species.types):
        if i < len(species.moves):
            species.moves[i].pkm_type = t

    n_moves = len(species.moves)

    # Randomly select a profile: 0=Balanced, 1=Offensive, 2=Defensive
    profile = rng.choice(2)

    """if profile == 1:  # Offensive Sweeper
        # Focus on Speed (idx 5) and one Attack (idx 1 or 3)
        # We give 252 to Speed, 252 to a random Attack, and 6 to HP
        atk_idx = rng.choice([1, 3])
        ev_list = [0] * 6
        ev_list[0] = 6
        ev_list[atk_idx] = 252
        ev_list[5] = 252
        evs = tuple(ev_list)"""
    if profile == 1:  # Defensive Tank
        # Focus on HP (idx 0) and one Defense (idx 2 or 4)
        def_idx = rng.choice([2, 4])
        ev_list = [0] * 6
        ev_list[0] = 252
        ev_list[def_idx] = 252
        ev_list[rng.choice([1, 3, 5])] = 6
        evs = tuple(ev_list)
    else:  # Balanced (Default)
        evs = tuple(list(int(x) for x in rng.multinomial(510, [1 / 6] * 6)))

    return Pokemon(
        species=species,
        move_indexes=list(sample([i for i in range(n_moves)], min(max_moves, n_moves))),
        level=100,
        ivs=(31,) * 6,
        evs=evs,
        nature=Nature(rng.choice(len(Nature), 1)[0]))


def gen_team(n: int,
             n_moves: int,
             rng: Generator = _RNG,
             _gen_move_set: MoveSetGenerator = gen_move_set,
             _gen_pkm_species: PokemonSpeciesGenerator = gen_pkm_species,
             _gen_pkm: PokemonGenerator = gen_pkm) -> Team:
    return Team([_gen_pkm(_gen_pkm_species(_gen_move_set(n_moves, rng), n_moves, rng), n_moves, rng) for _ in range(n)])


def generate_test_setup():
    # 1. Specialized Moves (STAB only)
    fire_move = Move(pkm_type=Type.FIRE, base_power=90, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="Flamethrower")
    water_move = Move(pkm_type=Type.WATER, base_power=90, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="Surf")
    grass_move = Move(pkm_type=Type.GRASS, base_power=90, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="Energy Ball")

    # 2. Species with polarized base stats
    fire_species = PokemonSpecies(base_stats=(80, 50, 60, 120, 60, 110), types=[Type.FIRE], moves=[fire_move], name="FireSweeper")
    water_species = PokemonSpecies(base_stats=(120, 50, 100, 70, 100, 50), types=[Type.WATER], moves=[water_move], name="WaterTank")
    grass_species = PokemonSpecies(base_stats=(100, 50, 90, 90, 90, 70), types=[Type.GRASS], moves=[grass_move], name="GrassBruiser")

    # 3. Team A: 1 Fire (Lead) and 2 Grass (Reserve)
    # The Fire lead will face a Water type and be forced to switch to Grass.
    our_team = Team([
        Pokemon(species=fire_species, move_indexes=[0], evs=(252, 0, 128, 0, 128, 0)), # Fire Lead
        #Pokemon(species=fire_species, move_indexes=[0], evs=(252, 0, 128, 0, 128, 0)), # Grass Reserve 1
        Pokemon(species=grass_species, move_indexes=[0], evs=(252, 0, 128, 0, 128, 0))  # Grass Reserve 2
    ])

    # 4. Team B: 3 Water types
    # This creates a "Bad Matchup" for our lead 100% of the time.
    opp_team = Team([
        Pokemon(species=water_species, move_indexes=[0], evs=(0, 0, 0, 252, 0, 252)),
        #Pokemon(species=water_species, move_indexes=[0], evs=(0, 0, 0, 252, 0, 252)),
        #Pokemon(species=grass_species, move_indexes=[0], evs=(0, 0, 0, 252, 0, 252))
    ])

    return our_team, opp_team
