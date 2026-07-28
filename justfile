# justfile — https://github.com/casey/just
# Instalar: pacman -S just (Arch) / brew install just (Mac) / veja o link acima
# Rodar uma receita: just <nome>, ex: just test

# lista as receitas disponíveis (roda ao chamar "just" sem argumento)
default:
    @just --list

# cria o venv e instala as dependências
install:
    python -m venv venv
    ./venv/bin/pip install -r requirements.txt

# roda a suíte de testes (mesma validação que o CI faz)
test:
    python -m pytest -v

# abre o notebook de análise
notebook:
    jupyter notebook notebooks/analysis.ipynb

# limpa caches do Python e do pytest
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .pytest_cache
