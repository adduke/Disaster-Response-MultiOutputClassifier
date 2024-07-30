# Disaster Response Project

Project contains all files to classify disaster response messages.

<p align="center">
  <a href=https://www.google.com/url?sa=i&url=https%3A%2F%2Fblogs.egu.eu%2Fdivisions%2Fnh%2F2021%2F06%2F&psig=AOvVaw1ITC4tybqI-iZCl1wCaVX6&ust=1722348290727000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCPjHsse1zIcDFQAAAAAdAAAAABAR>
    <img src="https://blogs.egu.eu/divisions/nh/files/2021/06/3-1-1000x1000.png" alt="Distaster response illustration" width="200" height="165">
  </a>
</p>

<h3 align="center">Disaster Response Classification</h3>

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

All packages required are in the requirements.txt.




## Project Motivation<a name="motivation"></a>

#### Project Motivation
When disasters strike, there’s often a surge of messages from various sources—tweets, texts, news reports, and more. For relief organizations, sifting through these messages and figuring out what needs immediate attention can be overwhelming.

#### The Challenge
Sorting through thousands of disaster-related messages manually isn’t just time-consuming; it can also lead to delays in getting help to where it’s needed. This is where automated systems can step in to make a difference.


#### What We’re Trying to Do
The Disaster-Response-MultiOutputClassifier project is about making the process a bit easier and more efficient. Our goals include:

1) Improved Categorization: We’re working on a system to help automatically sort messages into various categories, which can streamline how they’re handled.

2) Handling Volume: By building a tool that can manage a large number of messages, we aim to support responders in keeping up with the influx during a crisis.

3) Seamless Integration: We hope to create a tool that can be easily integrated into existing systems, helping teams work more effectively.


#### Why It Matters
Our hope is that by automating the categorization of messages, we can assist relief organizations in managing their workload. This way, they can focus more on providing the necessary support and aid where it’s most needed.




## File Descriptions <a name="files"></a>

There are 2 scripts available here to showcase work related to the above questions; process_data.py and train_classifier.py. Markdown cells were used to assist in walking through the thought process for individual steps.  

The main file to is the web app, simply run in terminal 'run.py' that allows the user to create a disaster message that can be classified by the model.

The notebooks folder was used for all necessary development of our model.
```
Disaster-Response-MultiOutputClassifier
│
├── data
│ ├── disaster_categories.csv
│ ├── disaster_messages.csv
│ ├── Twitter-sentiment-self-drive-DFE.csv
│ ├── test_engine.ipynb
│ └── process_data.py
│
├── models
│ ├── train_classifier.py
│ ├── classifier.pkl
│ └── model.py
│
├── app
│ ├── run.py
│ └── templates
│ ├── index.html
│ └── go.html
│
├── notebooks
│ ├── Exploration.ipynb
│ └── Modeling.ipynb
│
├── tests
│ ├── test_app.py
│ └── test_model.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
```



## Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage




## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Appen.com/ for the data.  You can find the Licensing for the data and other descriptive information at link available [here](https://www.appen.com/).  Otherwise, feel free to use the code here as you would like! 