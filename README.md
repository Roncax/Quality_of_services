# QoS_ML

The main goal of this competition is to predict the satisfaction of a customer of a mobile cellular network operator starting from network measurements only. A customer may be satisfied (0) or unsatisfied (1) with the network service: the operator is interested in predicting the satisfaction grade from network measurements in order to take action to avoid that unsatisfied customers change operator.

Students are required to download the training data and train machine learning classifier to predict the user satisfaction level. Students are encouraged in doing the following:

    focus on classifier algorithms, starting from simple ones and moving to more complex ones
    perform feature engineering and selection, adding new features, transforming existing ones or removing those who don't bring knowledge (hint: what could be computed started from volume of data downloaded and time taken for the download?)
    Use your daily amount of submission wisely! Upload a solution when you have tested on your laptops that it may improve your score!


File descriptions

    train_data.csv - the training set. Contains network measurements and associated user satisfaction.
    test_data.csv - the test set. Contains network measurements only. You have to predict the corresponding satisfaction level.
    sampleSubmission.csv - a sample submission file in the correct format

Data fields

The train_data.csv file contains the following columns:

    User_Id - an anonymous id unique to a given customer of the cellular network
    Cumulative_YoutubeSess_LTE_DL_Time - the total time in seconds the user spent downloading YouTube content with LTE (4G) technology
    Cumulative_YoutubeSess_LTE_DL_Volume - the total YouTube data volume in kBytes downloaded by the user with LTE (4G) technology
    Cumulative_YoutubeSess_UMTS_DL_Time - the total time in seconds the user spent downloading YouTube content with UMTS (3G) technology
    Cumulative_YoutubeSess_UMTS_DL_Volume - the total YouTube data volume in kBytes downloaded by the user with UMTS (3G) technology
    Max_RSRQ - The maximum Reference Signal Received Quality reported by the user in dB
    Max_SNR - The maximum Signal to Noise Ratio received by the user in dB
    Cumulative_Full_Service_Time_UMTS - the total time in seconds the user reported full UMTS service
    Cumulative_Lim_Service_Time_UMTS - the total time in seconds the user reported limited UMTS service (emergency calls only)
    Cumulative_No_Service_Time_UMTS - the total time in seconds the user reported no UMTS service (no network signal)
    Cumulative_Full_Service_Time_LTE - the total time in seconds the user reported full LTE service
    Cumulative_Lim_Service_Time_LTE - the total time in seconds the user reported limited LTE service (emergency calls only)
    Cumulative_No_Service_Time_LTE - the total time in seconds the user reported no LTE service (no network signal)
    User_Satisfaction - the satisfaction level reported by the user through a questionnaire. 0 = satisfied, 1 = unsatisfied.
