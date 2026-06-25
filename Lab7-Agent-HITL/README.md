# LAB 7 : Agent HITL (Human-In-The-Loop)

## Description

Ce laboratoire présente la mise en œuvre d'un agent HITL (Human-In-The-Loop) avec LangChain et LangGraph.

Un agent HITL est un agent d'intelligence artificielle qui interrompt son exécution afin de demander une validation, une correction ou une décision humaine avant de poursuivre.

## Contenu du TP

### Partie 1 : Définition des outils

- `read_email` : lit le contenu d'un e-mail depuis l'état de l'agent.
- `send_email` : simule l'envoi d'un e-mail.

### Partie 2 : Création d'un agent HITL

Création d'un agent utilisant :

- `create_agent`
- `HumanInTheLoopMiddleware`
- `InMemorySaver`

Le middleware permet d'intercepter l'exécution de l'outil `send_email` afin de demander une validation humaine.

### Partie 3 : Approuver le résultat

Validation de l'action proposée par l'agent avec :

```python
{"type": "approve"}
