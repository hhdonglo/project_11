# Puls-Events RAG — Assistant pour la recommandation d'evenements culturels

#GitHub
https://github.com/hhdonglo/project_11.git


Systeme de Generation Augmentee par Recuperation (RAG) pour recommander des evenements culturels a Paris, developpe avec LangChain, Mistral AI et FAISS.

**Projet :** OpenClassrooms Projet 11, Concevoir et deployer un systeme RAG
**Contexte de mission :** Puls-Events, Jeremy comme responsable technique
**Auteur :** Hope Donglo

---

## Presentation

J'ai realise ce projet pour repondre a une question simple. Est-ce qu'un chatbot peut recommander des evenements culturels a Paris a partir de vraies donnees, sans inventer d'informations.

Un assistant IA generique ne connait pas les evenements locaux recents. Il risque d'inventer des reponses plausibles mais fausses. J'ai donc construit un systeme RAG complet pour resoudre ce probleme.

Voici les etapes de mon pipeline.

1. Je recupere les evenements culturels en direct depuis l'API Open Agenda.
2. Je nettoie et filtre ces evenements pour ne garder que ceux des douze derniers mois, situes a Paris.
3. Je decoupe et vectorise ces evenements avec le modele mistral-embed.
4. J'indexe les vecteurs dans FAISS.
5. Je sers une interface conversationnelle avec Streamlit. Cette interface s'appuie sur une chaine LangChain qui recupere les evenements pertinents et genere des reponses en francais, ancrees dans les donnees reelles.

La chaine LangChain qui alimente le chatbot en direct est exactement la meme que celle evaluee par RAGAS. Je n'ai pas duplique la logique de recuperation et de generation ailleurs dans le code.

---

## Structure du projet

```
puls-events-rag/
├── src/
│   ├── ingestion/
│   │   └── fetch_events.py       # Client API Open Agenda, nettoyage, fetch_metadata.json
│   ├── processing/
│   │   ├── chunker.py            # Construit un chunk enrichi par evenement
│   │   └── embedder.py           # Envoie les evenements par lots vers mistral-embed
│   ├── vector_store/
│   │   └── build_index.py        # Construit l'index FAISS et metadata.json
│   ├── rag/
│   │   ├── langchain_chain.py    # Chaine RAG canonique (LangChain + Mistral + FAISS)
│   │   └── evaluate.py           # Jeu de test annote et evaluation RAGAS
│   └── app/
│       └── chatbot.py            # Interface de chat Streamlit (importe langchain_chain.ask)
├── tests/
│   └── test_data_pipeline.py     # 25 tests unitaires pytest
├── notebooks/
│   └── 01 a 04                   # Notebooks d'exploration (donnees, embedding, index, RAG)
├── data/
│   ├── raw/                      # Evenements bruts et nettoyes, fetch_metadata.json (non versionnes)
│   └── processed/                # Chunks, embeddings, index FAISS, resultats d'evaluation
├── reports/                      # Rapport technique (Word/PDF)
├── slides/                       # Presentation (PowerPoint, FR et EN)
├── 01_fetch.sh                   # Scripts d'etape du pipeline (autonomes : ecrivent et executent)
├── 02_embed.sh
├── 03_index.sh
├── 04_run.sh
├── 05_test.sh
├── 06_evaluate.sh
├── 07_langchain_integration.sh
├── pipeline.sh                   # Script maitre, execute les 7 etapes
├── .env.example                  # Modele de variables d'environnement
└── pyproject.toml                # Gestion des dependances avec Poetry
```

---

## Installation

### Prerequis

J'utilise Python 3.13. La contrainte est bornee a `>=3.13,<3.15` car faiss-cpu l'exige.

J'utilise Poetry pour gerer les dependances. La documentation d'installation est disponible sur python-poetry.org.

Il faut aussi une cle API Mistral avec un niveau de facturation actif, disponible sur console.mistral.ai. Et une cle API Open Agenda, disponible sur openagenda.com.

### Mise en place

```bash
git clone <repo-url>
cd puls-events-rag

poetry install
poetry add langchain-community

cp .env.example .env
```

J'edite ensuite le fichier .env avec mes propres cles.

### Contenu du fichier .env

```
MISTRAL_API_KEY=votre_cle_mistral
OPENAGENDA_API_KEY=votre_cle_openagenda
AGENDA_UID=82290100
MAX_EVENT_AGE_DAYS=365
TOP_K_RESULTS=5
```

---

## Utilisation

### Tout executer d'un coup

```bash
bash pipeline.sh --skip-run
```

Cette commande execute les 7 etapes du pipeline sans lancer le chatbot a la fin. Chaque etape est ignoree automatiquement si son fichier de sortie existe deja. J'ajoute l'option --force quand je veux tout reconstruire depuis zero.

### Lancer la demonstration en direct

```bash
bash pipeline.sh
```

Ou, si les donnees et l'index existent deja :

```bash
bash 04_run.sh
```

Le chatbot s'ouvre sur localhost:8501.

### Executer les etapes une par une

| Etape | Commande | Produit |
|---|---|---|
| 1. Recuperer les evenements | bash 01_fetch.sh --force | data/raw/events_paris.json, data/raw/fetch_metadata.json |
| 2. Decouper et vectoriser | bash 02_embed.sh --force | data/processed/chunks.json, data/processed/embeddings.npy |
| 3. Construire l'index FAISS | bash 03_index.sh --force | data/processed/faiss_index.idx, data/processed/metadata.json |
| 4. Lancer les tests unitaires | bash 05_test.sh | resultats pytest, 25 tests |
| 5. Verifier la chaine LangChain | bash 07_langchain_integration.sh | src/rag/langchain_chain.py, test de fumee |
| 6. Lancer l'evaluation | bash 06_evaluate.sh | data/processed/test_dataset.csv, data/processed/ragas_results.csv |
| 7. Lancer le chatbot | bash 04_run.sh | application Streamlit sur localhost:8501 |

### Options du pipeline

```bash
bash pipeline.sh --force
bash pipeline.sh --skip-run
bash pipeline.sh --skip-test
bash pipeline.sh --skip-langchain
bash pipeline.sh --skip-eval
```

L'option --force reconstruit chaque etape depuis zero. L'option --skip-run execute tout sauf le lancement du chatbot. L'option --skip-test saute la validation par les tests unitaires. L'option --skip-langchain saute la verification LangChain, mais elle bloque si j'essaie de lancer l'evaluation ou le chatbot en meme temps, puisque les deux en dependent. L'option --skip-eval saute l'evaluation RAGAS.

---

## Architecture

Voici le flux general de mon systeme.

L'API Open Agenda fournit les evenements bruts. Je les nettoie et je les filtre sur douze mois et sur la region parisienne. Je les decoupe en chunks. J'envoie ces chunks a mistral-embed pour obtenir des vecteurs.

Quand un utilisateur pose une question, celle-ci passe par src/rag/langchain_chain.py. Cette chaine LangChain recupere les evenements pertinents puis genere une reponse via le modele Mistral. Cette meme chaine sert deux usages differents. Le chatbot Streamlit l'utilise pour repondre en direct. Le script d'evaluation l'utilise pour tester 30 questions annotees avec RAGAS.

Le module langchain_chain.py enveloppe l'index FAISS dans la classe FAISS de LangChain. J'utilise ChatMistralAI pour la generation. J'assemble la recuperation et la generation avec le langage d'expression LangChain, aussi appele LCEL. Le chatbot et le script d'evaluation appellent tous les deux la fonction ask() de ce module. Je n'ai pas de logique dupliquee ailleurs dans le code.

---

## Qualite des donnees et tests

Je valide que chaque evenement a moins de douze mois par rapport a la date de recuperation, et non par rapport a la date d'execution des tests. Cette date de recuperation est enregistree dans data/raw/fetch_metadata.json. Cette approche garde la suite de tests stable, meme si je l'execute plusieurs semaines apres avoir recupere les donnees.

Les requetes reseau pendant la recuperation utilisent une logique de nouvelle tentative avec delai exponentiel. J'ai deux niveaux de protection. Le premier niveau gere les erreurs HTTP standard via urllib3. Le second niveau gere les coupures de connexion qui surviennent pendant la lecture de la reponse, un probleme que j'ai rencontre sur l'API Open Agenda lors de longues recuperations paginees.

Les appels a l'API Mistral, que ce soit pour les embeddings ou pour la generation, passent aussi par une logique de nouvelle tentative. Cette logique gere les erreurs de limitation de debit.

J'ai ecrit 25 tests automatises avec pytest. Ils s'executent avant chaque lancement du chatbot.

| Groupe de tests | Nombre | Ce qu'il verifie |
|---|---|---|
| TestRawData | 6 | Presence du fichier, titres et descriptions non nuls, dates presentes, metadonnees de recuperation |
| TestDateFilter | 3 | Recence sur douze mois par rapport a la date de recuperation, absence d'evenements trop lointains |
| TestLocationFilter | 3 | Au moins 90% des evenements a Paris, absence de valeurs de ville nulles ou mal formees |
| TestVectorDatabase | 10 | Dimension de l'index, synchronisation index et metadonnees, forme et type des embeddings, validite de la recherche |
| TestChunkQuality | 3 | Texte de chunk non vide, marqueurs requis presents, URL valides |

---

## Pile technologique

| Composant | Technologie |
|---|---|
| Modele d'embedding | mistral-embed, 1024 dimensions |
| Modele de generation | mistral-small-latest |
| Orchestration | LangChain, chaine LCEL |
| Base de donnees vectorielle | FAISS, IndexFlatL2 |
| Interface | Streamlit |
| Source de donnees | API Open Agenda v2 |
| Evaluation | RAGAS |
| Tests | pytest |
| Gestion des dependances | Poetry |
| Langage | Python 3.13 |

---

## Limites actuelles du POC

Je liste ici les limites que je connais pour cette version de preuve de concept.

Je n'utilise qu'un seul agenda Open Agenda comme source, celui de Deciding for Paris, avec l'identifiant 82290100. J'execute le pipeline manuellement, il n'y a pas de rafraichissement programme. J'utilise FAISS IndexFlatL2, qui fait une recherche exacte mais qui ne convient pas au dela d'environ 100 000 vecteurs. Je n'injecte pas l'historique de conversation dans le contexte du RAG, chaque question est traitee independamment. Ma base de connaissances reste statique entre deux reconstructions manuelles.

Le rapport technique detaille mes recommandations pour la version de production.

---

## Livrables

| Livrable | Emplacement |
|---|---|
| README, ce document | README.md |
| Gestion des dependances | pyproject.toml, poetry.lock |
| Scripts de pre-traitement et de vectorisation, avec docstrings | src/ingestion/, src/processing/, src/vector_store/ |
| Tests unitaires integres au pipeline | tests/test_data_pipeline.py, executes automatiquement dans pipeline.sh |
| Code du systeme RAG, LangChain + Mistral + FAISS | src/rag/langchain_chain.py |
| Rapport technique | reports/ |
| Presentation, 10 a 15 diapositives, en francais et en anglais | slides/ |
| Demonstration en direct | bash 04_run.sh |
