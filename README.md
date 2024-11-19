# Agentic Coach: Hackweek

An Agentic Coach for Lattice Hackweek.

[Proposal here](https://www.notion.so/lattice/Hackweek-Lattice-Coach-13e7372d6085800d866ac757462d8ca4?pvs=4)

## Mocks

- Lattice endpoints will be mocked for expediency

## Integration

Github
Gcal

## Running the app

### Daily Development

Every time you open a new terminal to work on this project:

```bash
# On Unix/MacOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate

# You can also run this to see the activation command:
make activate
```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Poetry (Python package manager)

### Installing Poetry for package management

Install Poetry by running:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Project Setup

1. Clone the repository

```bash
git clone <repository-url>
cd agentic_coach
```

2. Install dependencies

```bash
poetry install
```

3. Activate the virtual environment

```bash
poetry shell
```

### Running the Project

```bash
make run
```

(or...)

```bash
poetry run python -m src
```

### Adding New Dependencies

To add a new package:

```bash
poetry add package-name
```

To add a development dependency:

```bash
poetry add --group dev package-name
```
