![image](https://user-images.githubusercontent.com/86326159/206014015-a70e3581-e15c-4a10-95ef-36fd5a560717.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

This set of notebooks goes through 4 steps: Data Preparation, Analysis (NLP), Post Processing, and a Dashboard to extract insights from natural language reviews on Steam.

1. <i>Data Preparation:</i> This notebook connects to the Steam API and returns a DataFrame with reviews and data about the author.
2. <i>Analysis:</i> This notebook runs a NLP pipeline to extract topics and sentiments. This notebook could be replaced with a different algorithm as well.
* In order to run this successfully, install this Maven package on the cluster: *com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0*
* You can do this by opening your Cluster > Libraries Tab > Install New > Library Source: Maven > Paste the coordinates: com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0
* Note: If you use the Solution Accelerator cluster, it takes care of this for you.
3. <i>Post Processing:</i> This notebook profiles and clusters the review authors, saving the results to a Delta table to be used in the dashboard.
4. <i>Player Feedback Sentiment Analysis Dashboard</i>: The dashboard uses [Databricks Lakeview Dashboards](https://docs.databricks.com/en/dashboards/lakeview.html) and is provided in this bundle as a json file. In order to use it, you can import the dashboard by going to Dashboards > Lakeview Dashboards > Hit the down arrow next to "Create Lakeview Dashboard" > Import Lakeview Dashboard from File 

___
<eduardo.brasileiro@databricks.com> <br>
<anil.joshi@databricks.com> <br>
<duncan.davis@databricks.com> <br>
<dan.morris@databricks.com> <br><br>

___

&copy; 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
