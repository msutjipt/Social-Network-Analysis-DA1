# Social-Network-Analysis-DA1

## 1. Background and Task Description
This project is based on a case study as part of the course Data Analytics 1 in the Master of Information Systems at the 
University of Muenster. The main task is to build analyze social network data from various perspectives such as the user, 
the social media posts or the relationships between them. 

## 2. Folder Structure 
```
└── Social-Network-Analysis-DA1
    ├── GROUP_F
    │   ├── 1_descriptive_analysis
    │   │   └── desriptive_analysis.ipynb
    │   ├── 2_community_detection
    │   │   └── community_detection.ipynb
    │   ├── 3_popularity_metrics
    │   │   └── popularity_metrics.ipynb
    │   ├── 4_sentiment_analysis
    │   │   └── sentiment_analysis.ipynb
    │   ├── 5_text_clustering
    │   │   └── text_clustering.ipynb
    │   ├── data
    │   ├── docs
    │   ├── models
    │   └── utils
    │       ├── __init__.py
    │       └── json_to_csv.py
    ├── README.md
    ├── .ruff.toml
    ├── poetry.lock
    └── pyproject.toml
```

## 3. Sections
The code for our project can be found in 3.1 - 3.5. 
A description of each section can be found below

### 3.1 1_descriptive_analysis
This folder contains a notebook covering basic descriptive 
analysis about the network structure and the users (nodes).

### 3.2 2_community_detection
This folder contains a notebook covering functionality for community detection. 
In particular, techniques based on the network representation as well as 
embeddings based on the graph.csv.

### 3.3 3_popularity_metrics
This folder contains a notebook covering functionality for calculating 
common popularity metrics such as page_rank, degree etc.

### 3.4 4_sentiment_analysis
This folder contains a notebook covering functionality for 
conducting a sentiment analysis based on a pre-trained 
transformer model from Huggingface.


### 3.5 5_text_clustering
This folder contains a notebook covering functionality for 
conducting a text clustering based on the posts from the users.

### 3.6 data
This folder contains all data, which was either given or produced during the 
case study 

### 3.7 docs 
This folder contains all figures, which we produced during the case study

### 3.8 docs 
This folder contains a trained node2vec model used in section 3.2


## 3. Getting Started
For dependency management we used Poetry and for code linting 
we used Ruff as external tools.

### 3.1 Install pipx 
To make use of Poetry you need to install pipx. This can be done by the following command 
```
pip install pipx
```

### 3.2 Installing Poetry 
After that you can install Poetry by typing:
```
pipx install poetry 
```

### 3.3 Install Poetry Virtual Machine
After that you can install the poetry virtual machine by typing
```
cd ./Social-Network-Analysis-DA1
```

```
poetry install
```

### 3.4 Using Poetry Virtual Machine
On Windows the default path for the poetry virtual machine is: 
```
C:\Users\your_user\AppData\Local\pypoetry\virtualenvs\
```

On MacOS/Linux the default path for the poetry virtual machine is: 
```
/Users/your_user/Library/Caches/pypoetry/virtualenvs
```