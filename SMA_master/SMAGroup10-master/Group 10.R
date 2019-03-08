
system("java -version")

.libPaths()
.libPaths("C:/Users/dtuiran/AppData/Local/Temp/RtmpSaJsE9/downloaded_packages")
.libPaths()


#install the required packages

if(!require("rtweet")) install.packages("rtweet"); library("rtweet")
if(!require("rlist")) install.packages("rlist"); library("rlist")
if(!require("data.table")) install.packages("data.table"); library("data.table")
if (!require("randomForest")) install.packages("randomForest", quiet=TRUE) ; require("randomForest")
if (!require("ROCR")) install.packages("ROCR", quiet=TRUE) ; require("ROCR")
if (!require("jsonlite")) install.packages("jsonlite", quiet=TRUE) ; require("jsonlite")
if (!require("rJava")) install.packages("rJava", quiet=TRUE) ; require("rJava")
if (!require("RWeka")) install.packages("RWeka", quiet=TRUE) ; require("RWeka")
if (!require("stringi")) install.packages("stringi", quiet=TRUE) ; require("stringi")
if (!require("tidyverse")) install.packages("tidyverse", quiet=TRUE) ; require("tidyverse")
if (!require("tidytext")) install.packages("tidytext", quiet=TRUE) ; require("tidytext")
if (!require("caret")) install.packages("caret", quiet=TRUE) ; require("caret")
if (!require("e1071")) install.packages("e1071", quiet=TRUE) ; require("e1071")
if (!require("SDMTools")) install.packages("SDMTools", quiet=TRUE) ; require("SDMTools")
if (!require("ggplot2")) install.packages("ggplot2", quiet=TRUE) ; require("ggplot2")
if (!require("caTools")) install.packages("caTools", quiet=TRUE) ; require("caTools")


for (i in c('SnowballC','slam','tm','RWeka','Matrix')){
  if (!require(i, character.only=TRUE)) install.packages(i, repos = "http://cran.us.r-project.org")
  require(i, character.only=TRUE)
}

if (!require("wordcloud")) {
  install.packages("wordcloud",repos="https://cran.rstudio.com/",
                   quiet=TRUE)
  require("wordcloud")
}

#Connecting to Twitter API to extract the data

appname <- "SMA_harshit"

#Autentication to connect to twitter API
consumer_key <- "POKDTELDvpHgCJPojeRmCqvT9"
consumer_secret <- "SNLnQfEYfkWbSEuf3qOljhysY8DodYdz88himGFPEIUFxbxJjh"
access_token <- "284556547-7kOWjMLa03uoWc4zcW6xsnU5Mk9jh3Q82MxNhzzY"
access_secret <- "zYKAwg2iZwiOv2nHwSTgCNQUFRL73xPZhUNJso8DTng7D"


twitter_token <- create_token(
  app = appname,
  consumer_key = consumer_key,
  consumer_secret = consumer_secret,
  access_token = access_token,
  access_secret = access_secret)

#Extracting tweets related to keyword tesla.

tesla_tweets <- rtweet::search_tweets(q = "tesla",
                                     n = 50000,include_rts = FALSE)



# Saving on object in RData format


save(tesla_tweets, file= "C:/Users/dtuiran/Documents/SMA/tesla_tweets.RData")

#Loading tweets previously selected and labeled tweets
load("C:/Users/dtuiran/Documents/SMA/tesla_tweets.RData")
labeled_tweets<- read.csv("C:/Users/dtuiran/Documents/SMA/Labeled_tweets.csv")

########################################
### Basic Text processing  #
########################################

#Removing labeled tweets from the entire data set of tweets
train_rows <-labeled_tweets[,1]

tesla_tweets<-tesla_tweets[-train_rows,]

#keeping only columns of interest
tesla_tweets<- tesla_tweets[,c("user_id","created_at","text","followers_count","statuses_count","location", "favorite_count","retweet_count")]

##################################################################################################################
### Pre-processing the entire tweets data set
##################################################################################################################

#Deleting tweets in a language different to English
del<-stri_enc_isascii(tesla_tweets$text)
tesla_tweets<-tesla_tweets[del,]

# Deleting non-recognizable characters, otherwise the tm package will get in trouble later. 
tesla_tweetsText <- sapply(tesla_tweets[,3],function(x) iconv(x, 'utf8', 'ascii',""))


#Reading the data into a corpus
tweets_corpus <- VCorpus(VectorSource(tesla_tweetsText))

#saving the extracted tweets in a csv file
tweets_csv <-data.frame(text = sapply(tweets_corpus, as.character), stringsAsFactors = FALSE)
#write.csv(tweets_csv,file="C:/Users/dtuiran/Documents/SMA/tweets_csv.csv")


#transformming words to lower case to make them standard
tweets_corpus <- tm_map(tweets_corpus,content_transformer(tolower))

#Remove stopwords
forremoval <- stopwords('english')
forremoval <-append(forremoval,c('the','tesla','tsla','teslas','will','elon','musk'))
forremoval

tweets_corpus <- tm_map(tweets_corpus, removeWords,c(forremoval)) 

#Delete users name y @ sign.
gsubtransfo <- content_transformer(function(x,from, to) gsub(from, to, x))
tweets_corpus <- tm_map(tweets_corpus, gsubtransfo, "@\\w+",  "")

#Delete URLS 
gsubtransfoURL <- content_transformer(function(x,from, to) gsub(from, to, x))
tweets_corpus <- tm_map(tweets_corpus, gsubtransfoURL, "\\s?(f|ht)(tp)(s?)(://)([^\\.]*)[\\.|/](\\S*)",  "")

#Remove numbers, punctuations and white spaces
tweets_corpus <- tm_map(tweets_corpus, removeNumbers)
tweets_corpus <- tm_map(tweets_corpus, stripWhitespace)
tweets_corpus<- tm_map(tweets_corpus, removePunctuation)

#Saving procesed tweets as a data frame
dataframetweets <- data.frame(text = sapply(tweets_corpus, as.character), stringsAsFactors = FALSE)
dataframetweets<-na.omit(dataframetweets)



# make a dtm with the cleaned-up corpus
dtmTweets <- DocumentTermMatrix(tweets_corpus, control = list( wordlengths=c(2,Inf),
                                                                 weighting =function(x) weightTf(x)))
inspect(dtmTweets)

#removen sparse termns and making the dtm dense
dtmTweetsDense <- removeSparseTerms(dtmTweets,0.998)
inspect(dtmTweetsDense)

#save the document term matrix as a data frame
dtm_TweetsDense<- as.data.frame(as.matrix(dtmTweetsDense))
colnames(dtm_TweetsDense) <- make.names(colnames(dtm_TweetsDense))


##################################################################################################################
### Pre-processing the  labeled tweets
##################################################################################################################

#Save column with labels as an independent vector
Ltweets <- labeled_tweets[,c(2,3)]
sentiment <- Ltweets[,2]

#Checking we have the same amount of labels for each sentiment class
table(Ltweets$Sentiment)

#Converting to ascii format 
Ltweets <- sapply(Ltweets$text,function(x) iconv(x, 'utf8', 'ascii',""))

#Save labeled tweets as a corpus
Ltweets_corpus <- VCorpus(VectorSource(Ltweets))

#transformming words to lower case to make them standard
Ltweets_corpus <- tm_map(Ltweets_corpus,content_transformer(tolower))

#Remove stopwords
Ltweets_corpus <- tm_map(Ltweets_corpus, removeWords,c(forremoval)) 

#Delete users name y @ sign.
gsubtransfo <- content_transformer(function(x,from, to) gsub(from, to, x))
Ltweets_corpus <- tm_map(Ltweets_corpus, gsubtransfo, "@\\w+",  "")

#Delete URLS
gsubtransfoURL <- content_transformer(function(x,from, to) gsub(from, to, x))
Ltweets_corpus <- tm_map(Ltweets_corpus, gsubtransfoURL, "\\s?(f|ht)(tp)(s?)(://)([^\\.]*)[\\.|/](\\S*)",  "")

#Remove number, spaces and puctuation
Ltweets_corpus <- tm_map(Ltweets_corpus, removeNumbers)
Ltweets_corpus <- tm_map(Ltweets_corpus, stripWhitespace)
Ltweets_corpus<- tm_map(Ltweets_corpus, removePunctuation)

#convert corpus to a data frame and adding labels column
dataframeLtweets <- data.frame(text = sapply(Ltweets_corpus, as.character), stringsAsFactors = FALSE)
dataframeLtweets <- cbind(dataframeLtweets, sentiment)
colnames(dataframeLtweets)[2] <- "cat"

#Make column with labels as a data frame
dataframeLtweets$cat<-as.factor(dataframeLtweets$cat)

# make a dtm with the cleaned-up corpus
#Output of the tm inspect() function summarizes the number of rows and terms in the matrix 
#and displays the first 10 rows and columns.

dtmLTweets <- DocumentTermMatrix(Ltweets_corpus, control = list( wordlengths=c(2,Inf),
                                                            weighting =function(x) weightTf(x)))

inspect(dtmLTweets)

#removen sparse termns and making the dtm dense
dtmLTweetsDense <- removeSparseTerms(dtmLTweets,0.995)
inspect(dtmLTweetsDense)

#save the document term matrix as a data frame
dtm_LTweetsDense<- as.data.frame(as.matrix(dtmLTweetsDense))
colnames(dtm_LTweetsDense) <- make.names(colnames(dtm_LTweetsDense))

# add the sentiment label
dtm_LTweetsDense$cat <- dataframeLtweets$cat


#####################################################################################################################################
## Inspect our text 
#####################################################################################################################################

# Word cloud based on the original text
tf <- termFreq(tesla_tweetsText)
wordcloud(names(tf),tf,
          max.words=50,
          scale=c(3,1),colors=brewer.pal(8, "Dark2"),random.color = TRUE)


# Word cloud based on the dtm matrix and all the conversions
m <- as.matrix(dtmLTweets)
#Count occurrences of each term
v <- sort(colSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
#create word cloud
options(warn=-1) #turn warnings off
wordcloud(d$word,d$freq,
          max.words=45,
          scale=c(3,1),colors=brewer.pal(8, "Dark2"),random.color = TRUE)


#Number of words per tweet
number_words<-as.data.frame(rowSums(dtm_LTweetsDense[sapply(dtm_LTweetsDense, is.numeric)], na.rm = TRUE))
number_words<-cbind(number_words,dtm_LTweetsDense$cat)
colnames(number_words)<-c("number_words","Sentiment")

#Number of word for positive tweets
qplot(number_words$number_words[number_words$Sentiment==-1], geom="histogram",binwidth=1,
      xlab="Number of words" , ylab="Count" ,fill=I("grey"), 
      col=I("black"), main = "Positive Tweets") 


#Number of word for negative tweets
qplot(number_words$number_words[number_words$Sentiment==1], geom="histogram",binwidth=1,
      xlab="Number of words" , ylab="Count" ,fill=I("grey"), 
      col=I("black"), main = "Negative Tweets") 

#Check the number of words per tweet for positive and negative sentiment
table(number_words)

#Spliting cleaned and labeled tweets in word tokens
tidy_corpus <- dataframeLtweets %>%
  unnest_tokens(word, text)

#Keep only words with more than 2 characters
tidy_corpus <-tidy_corpus[nchar(tidy_corpus$word) >= 2 , ]


#Words used more than 30 times
tidy_corpus %>%
  count(word, sort = TRUE) %>%
  filter(n > 30) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip() +
  ggtitle("Most used words")


tesla_tweets$Date <- as.Date(tesla_tweets$created_at)
tesla_tweets$Time <- format(tesla_tweets$created_at,"%H")
tesla_tweets$day <- weekdays(as.Date(tesla_tweets$Date))

#tweets by hour
tweets_time <- tesla_tweets %>%
  select(Time, text) %>%
  group_by(Time) %>%
  summarise(tweet_count = n())


ggplot(data=tweets_time,aes(Time, tweet_count)) +
  geom_col() +
  xlab(NULL) +
  coord_flip() 

####### Sentiment over the day

nrcmad <- get_sentiments("nrc") 

tidy_corpus %>%
  inner_join(nrcmad, by= c("word","word")) %>%
  count(word, sort = TRUE)


# sentiment of words in sms messages
tweets_word_counts <- tidy_corpus %>%
  inner_join(get_sentiments("bing"),by= c("word","word")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

tweets_word_counts


#NUMBER OF FAVORITES AND RETWEETS



qplot(tesla_tweets$favorite_count[tesla_tweets$favorite_count<15], geom="histogram",binwidth=1,
      xlab= "Number favorites",fill=I("grey"), 
      col=I("black"), main="Favorites distribution") 

qplot(tesla_tweets$retweet_count[tesla_tweets$retweet_count<15], geom="histogram",binwidth=1,
      xlab= "Number Retweets",fill=I("grey"), 
      col=I("black"), main="Retweets distribution") 

qplot(tesla_tweets$followers_count[tesla_tweets$followers_count<8000], geom="histogram",binwidth=500,
      xlab= ("Number of followers"),fill=I("grey"), 
      col=I("black"), main="Followers count") 


#NUMBER OF FOLLOWERS 
followers<-as.data.frame(subset(tesla_tweets$followers_count,!duplicated(tesla_tweets$user_id)))
colnames(followers)<-"Number_followers"


#Get NRC sentiment lexicons in a tidy format
nrcmad <- get_sentiments("nrc") 

tidy_corpus %>%
  inner_join(nrcmad, by= c("word","word")) %>%
  count(word, sort = TRUE)


# sentiment of words in tweets
tweets_word_counts <- tidy_corpus %>%
  inner_join(get_sentiments("bing"),by= c("word","word")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()
tweets_word_counts


#wordcloud negative and positive labeled tweets 
install.packages("tidytext")
library(tidytext)
install.packages("reshape")
library(reshape)
tidy_corpus %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("red", "blue"),
                   max.words = 100)

# plot the top_n word for each type of sentiment
tweets_word_counts %>%
  group_by(sentiment) %>%
  top_n(20) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment",
       x = NULL) +
  coord_flip()


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

#Build classification models for sentiment analysis
set.seed(123)
dtm_LTweetsDense_combined = cbind(dtm_LTweetsDense,dataframeLtweets)


# Build a training and testing set
split <- sample.split(dtm_LTweetsDense_combined, SplitRatio=0.7)
trainDenseL <- testDenseL <- NULL
trainDenseL <- subset(dtm_LTweetsDense_combined, split==TRUE)
testDenseL <- subset(dtm_LTweetsDense_combined, split==FALSE)

train <- subset(trainDenseL , select = -436:-434 )
test <-  subset(testDenseL , select = -436:-434 )

#  Random forest model.
RF_model_train <- randomForest(cat~., data=train,ntree=1699,maxnodes=100)
RF_predict <- predict(RF_model_train, newdata= test )
table(test$cat, RF_predict)

confMatrixRF <- confusionMatrix(RF_predict, test$cat, positive="1")
confMatrixRF



# Calculate auc for test data set
predRF <- prediction(as.numeric(RF_predict),as.numeric(test[,433]))


# ROC curve
perfRF <- performance(predRF,"tpr","fpr")
plot(perfRF,col="red")
abline(0,1)

## auc
auc.perfRF = performance(predRF, measure = "auc")
auc.perfRF@y.values


##### Applying the model on main dataset 
directory <- ("C:/Users/dtuiran/Documents/SMA")
dataframetweetsnew <- read.csv("dataframetweets2.csv")
dataframetweetsnew$doc_id = NULL
Mtweets <- VCorpus(VectorSource(dataframetweetsnew))
dtm_Mtweets <- DocumentTermMatrix(Mtweets, control = list( wordlengths=c(2,Inf),
                                                           weighting =function(x) weightTf(x)))


# convert to matrices for subsetting
train_validate <- dtm_LTweetsDense # training
test_validate <- dtm_Mtweets # testing

# subset testing data by colnames (ie. terms) or training data
interse = intersect(colnames(test_validate), colnames(train_validate))

#DTM of unlabeled tweets saved as dataframe
setwd(dir = "C:/Users/dtuiran/Documents/SMA")
file_test = read.csv("dataframeunlabeled.csv")
file_test$X = NULL
names(file_test) <- make.names(names(file_test))

### Predicting sentiments for final dataset 
RF_predict3 <- predict(RF_model_train, newdata=file_test)
RF_predict3 = as.character(RF_predict3)

dataframetweetsnew = cbind(dataframetweetsnew,RF_predict3)

head(dataframetweetsnew)




################################################################################################################# 
#################################################################################################################
#################################################################################################################
# SVM Model 

# model specification
SVM_model_train <- svm(cat~., data=trainDenseL,
                       type="C-classification",
                       kernel="radial")

# print a summary
SVM_model_train

#cross validation
#10 fold cross validation
#when cost is small, margins will be wide, more tolerance for missclasification
tune.out=tune(svm,cat~.,data=train,
              ranges=list(cost=c(0.01,0.1,1.5,10,100)))

#print results for cross validation
print(tune.out)


#Run the model with best parameters results
SVM_model_train <- svm(cat~., data=train,
                       type="C-classification",
                       kernel="radial",
                       cost= 100)

#Make predictions with the model using the test data set
SVM_predict <- predict(SVM_model_train,(test),type='prob')
table(test$cat, SVM_predict)

#Create a confusion matrix
confMatrix1 <- confusionMatrix(SVM_predict, test$cat, positive="1")
confMatrix1

# Calculate auc for test data set
predSVM <- prediction(as.numeric(SVM_predict),as.numeric(testDenseL[,433]))

# ROC curve
perfSVM <- performance(predSVM,"tpr","fpr")


## AUC 
auc.perfSVM = performance(predSVM, measure = "auc")
auc.perfSVM@y.values


#PLOT ROC curve for SVM and random forest models.
plot(perfSVM,col="blue")
par(new=TRUE)
plot(perfRF,col="red")
abline(0,1)

plot(SVM_predict)


### Predicting sentiments for final dataset 
RF_predict4 <- predict(SVM_model_train, newdata=file_test)
RF_predict4 = as.character(RF_predict4)
dataframetweetsnewSVM = cbind(dataframetweetsnew,RF_predict4)
head(dataframetweetsnew)

barplot(table(dataframetweetsnewSVM$RF_predict4))
