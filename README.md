# Trademark Search AI Bot
**Table of Contents**
- [Introduction](#header1)
- [Data Collection & Preparation](#header2)
- [Trademark Bot](#header3)
- [Next Steps](#header4)

<a id="header1"></a>
## Introduction

Our goal for this project was to create a ML-based search bot that would identify if a trademark name was available to be registered or not.  The bot would perform the following functions:
- Prompt the user for a trademark name
- Search the database of all available trademarks for the name, or any similar trademark names for potential conflicts
- Provide a summary of findings and an opinion on whether the prompted trademark can be registered, or if there is a conflict.

<a id="header2"></a>
## Data Collection and Preparation
Our data was sourced from the United States Patent and Trademark Office [website](https://www.uspto.gov/ip-policy/economic-research/research-datasets/trademark-case-files-dataset).  The data was set up with a multi-CSV structure that contained different aspects of the data.  This data was joined together by unique serial numbers attributed to specific trademark names.
![Data Schema](/Images/USPTO%20Trademark%20.png)

We focused on data from three tables:
- The case file data, which included the serial number, trademark name, and case_file status
- The classification data, which included the first use anywhere data for the trademarks in the database
- The statement data, which contained a brief description of the industry that the trademark name applies to.

We concatenated these tables together, and then removed duplicate rows as well as the null values in the mark_id_char (Trademark Name) column.  We also updated the column names to be more user-friendly.  Once that was done, we were ready to build our trademark bot.

<a id="header3"></a>
## Trademark Bot
We used langchain to build out our class framework, using a 'llama3' pre-trained model for our foundation, and then applying a sentence transformer from HuggingFace for the embeddings.  We then created functions to prompt the user for a trademark name, and then filter the data down to any trademark that contained all or part of the trademark (excluding stopwords).  Once the data was filtered, the bot would ingest the data and then return its results.

After updating and tweaking the System Prompt, the resulting Trademark Bot was able to correctly identify whether a trademark appeared in the filtered dataframe, and it would also provide a summary of the closest potential conflicts from other trademarks.

<a id="header4"></a>
## Next Steps
While we were excited with the results from our initial efforts, there is still room for improvement.  Using alternative models or embedding systems may help us to provide more detailed results.  

We would also like to be able to have the bot write more detailed summaries or blurbs on potential conflicting trademark names to provide the applicant for additioanl context.  Additionally, we would need to determine if there is an API that could access the data in the USPTO and provide automatic updates to the dataset with the most up-to-date trademark information so our bot could stay current.

