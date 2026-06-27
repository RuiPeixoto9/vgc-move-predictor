"""Constantes partilhadas — alinhado com reunião 2026-05-15 e notas do orientador.

Setup actual do dataset:
  - 2 pokémon por equipa, 4 moves cada  ->  [2, 4, +1 switch]  = 9 outputs
  - Augmentation:  X,Y -> Xm  (T0||T1, target moves T0)
                   Y,X -> Ym  (T1||T0, target moves T1)  [swap no encoding]
"""
from __future__ import annotations

N_POKEMON = 2
N_MOVES_PER_POKEMON = 4
N_MOVE_SLOTS = N_POKEMON * N_MOVES_PER_POKEMON  # 8
N_SWITCH_SLOT = 8
N_TARGETS = N_MOVE_SLOTS + 1  # 9

# Indices em Y: [pkm0: 0..3, pkm1: 4..7, switch: 8]
PKM0_SLICE = slice(0, 4)
PKM1_SLICE = slice(4, 8)
SWITCH_SLICE = slice(8, 9)
