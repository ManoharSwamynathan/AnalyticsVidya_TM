### Problem Statement (Text Mining Challenge)

Competition Source URL(Analytics Vidya): https://datahack.analyticsvidhya.com/contest/mlware-1/

Brevity is the soul of wit - William Shakespeare, Hamlet

Shakespeare probably saw a world expressing itself on Twitter before any one did! And he would have known if any one made a sarcastic tweet or not. But for our machines and bots, they need help from data scientists to help them decipher sarcasm.

Your ask for this competition is to build a machine learning model that, given a tweet, can classify it correctly as sarcastic or non-sarcastic. This problem is one of the most interesting problems in Natural Language Processing and Text Mining. You never know - the next generation of bots might come and thank you for making them smarter!

The prediction has to be made using only the text of the tweet.

#### Dataset
Two files - one each for training and testing - are provided.

training.csv - This file contains three columns -
* ID - ID for each tweet
* tweet - contains the text of the tweet
* label - the label for the tweet (‘sarcastic’ or ‘non-sarcastic’)

test.csv - This file has two columns containing the ID and tweets. The predictions on this set would be judged

#### Evaluation:

The metric used for evaluating the predictions for this problem is simply the F1-score.
Public : Private leaderboard split on test data is 25:75 

#### Results
RNN model using passage package scored 0.752787 on the private leaderboard
