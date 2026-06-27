## Dependência obrigatória: VGC AI Framework 2

```bash
git clone https://gitlab.com/DracoStriker/pokemon-vgc-engine.git
cd pokemon-vgc-engine
pip install .
```

## Setup

```bash
git clone <URL-deste-repositorio>
cd vgc-move-predictor
python -m venv .venv
source .venv/Scripts/activate   # Git Bash (Windows)
pip install -r requirements.txt

# Noutra pasta: instalar vgc2 (ver acima)
```

Colocar na pasta de trabalho (ou gera com os scripts):

- `move_count_sup_*.npz` - dataset de supervisão
- `move_predictor_local.pt` - checkpoint do previsor (não versionado no Git)
- `sup_guidelines_rules.pt` - agente IL para demos e rollouts (não versionado)

Ficheiros JSON de arquitetura (ex. `move_predictor_local.json`) podem ser guardados à parte; são recriados no treino.

## Pipeline principal

```bash
# 1) Dataset de supervisão (requer moves_analytics pré-gerado)
python -u build_move_count_supervision.py --glob moves_analytics_*.npz --out move_count_sup_switch.npz --concat_rules_to_x

# 2) Treino + avaliação
python -u run_move_predictor_pipeline.py --data move_count_sup_switch.npz

# 3) Validação KL (dataset)
python -u eval_move_predictor_kl.py --data move_count_sup_switch.npz --ckpt move_predictor_local.pt

# 4) Validação em simulação
python -u eval_move_predictor_sim.py --ckpt move_predictor_local.pt

# 5) Piloto GA (rule balance)
python -u balance_team_predictor_ga.py --predictor_ckpt move_predictor_local.pt
```

## Demonstração Godot

1. Abrir `pokemon-vgc-engine/visual_server` no Godot e carrega em Play (porta UDP 12345).
2. Nesta pasta:

```bash
python -u demo_godot_battle.py \
  --agent_ckpt sup_guidelines_rules.pt \
  --switch_agent_path battle_agent.py \
  --team_generator external --generator_path gen.py
```

## Ficheiros auxiliares

- `battle_agent.py` — política híbrida com exploit de switch
- `gen.py` — gerador externo de equipas

## Licença

Código do estágio sob responsabilidade do autor; `vgc2` segue a licença MIT do [pokemon-vgc-engine](https://gitlab.com/DracoStriker/pokemon-vgc-engine).
