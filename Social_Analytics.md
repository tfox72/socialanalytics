
## Social_Analytics - Distinguishing Sentiments


## News Mood

In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: __BBC, CBS, CNN, Fox, and New York times__.

![output_10_0.png](output_10_0.png)

![output_13_1.png](output_13_1.png)

The first plot will be and/or feature the following:

* Be a scatter plot of sentiments of the last __100__ tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
* Each plot point will reflect the _compound_ sentiment of a tweet.
* Sort each plot point by its relative timestamp.

The second plot will be a bar plot visualizing the _overall_ sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

* Pull last 100 tweets from each outlet.
* Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet. 
* Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
* Export the data in the DataFrame into a CSV file.
* Save PNG images for each plot.

As final considerations:

* Use the Matplotlib and Seaborn libraries.
* Include a written description of three observable trends based on the data. 
* Include proper labeling of your plots, including plot titles (with date of analysis) and axes labels.
* Include an exported markdown version of your Notebook called  `README.md` in your GitHub repository.  




```python
# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import tweepy
import API_Keys
import numpy as np
from datetime import datetime
```


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
consumer_key = API_Keys.Twitter_Key
consumer_secret = API_Keys.Twitter_Secret
access_token = API_Keys.Twitter_Token
access_token_secret = API_Keys.Twitter_Token_Secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
sentiment_df=[]
```


```python
# Target News Network
users = ("@BBC","@CBS","@CNN","@FoxNews","@nytimes") 
sentiment= []

```


```python


for user in users:
    try:
        #Get Tweets from feeds
        public_tweets = api.user_timeline(user,count = 100, result_type="recent")
        
        #Set counter
        counter = 100
            
        for tweet in public_tweets:
            text = tweet['text']
            time = tweet['created_at']
            date = datetime.strptime(time, '%a %b %d %H:%M:%S %z %Y').date()
            
        #Run Vader analysis
            scores = analyzer.polarity_scores(text)
                
            
        #Add Value to appropriate list
            scores['Data_Source'] = user
            scores['Date'] = date
            scores['tweet'] = counter
            counter -=1
                
                
            sentiment.append(scores)
                
            
    except tweepy.TweepError:
        print("Failed to run the command on that user, Skipping...")
    continue
sentiment_df = pd.DataFrame(sentiment)
sentiment_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Data_Source</th>
      <th>Date</th>
      <th>compound</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>2017-12-10</td>
      <td>0.6915</td>
      <td>0.000</td>
      <td>0.560</td>
      <td>0.440</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBC</td>
      <td>2017-12-10</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBC</td>
      <td>2017-12-10</td>
      <td>0.4391</td>
      <td>0.000</td>
      <td>0.855</td>
      <td>0.145</td>
      <td>98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBC</td>
      <td>2017-12-10</td>
      <td>-0.3818</td>
      <td>0.126</td>
      <td>0.874</td>
      <td>0.000</td>
      <td>97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBC</td>
      <td>2017-12-10</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>96</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivot_df = sentiment_df.pivot(index='tweet',columns='Data_Source',values='compound')
pivot_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Data_Source</th>
      <th>@BBC</th>
      <th>@CBS</th>
      <th>@CNN</th>
      <th>@FoxNews</th>
      <th>@nytimes</th>
    </tr>
    <tr>
      <th>tweet</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0772</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.4939</td>
      <td>0.6289</td>
      <td>0.4588</td>
      <td>0.0000</td>
      <td>0.7351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6808</td>
      <td>0.9476</td>
      <td>0.5574</td>
      <td>0.3182</td>
      <td>-0.5972</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>0.5106</td>
      <td>-0.4939</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>0.5826</td>
      <td>0.0516</td>
      <td>-0.4019</td>
      <td>-0.5423</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.5106</td>
      <td>0.4201</td>
      <td>-0.3182</td>
      <td>-0.0772</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.4588</td>
      <td>0.6696</td>
      <td>-0.0253</td>
      <td>0.0000</td>
      <td>-0.2960</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.4767</td>
      <td>0.6696</td>
      <td>0.0000</td>
      <td>-0.6597</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0000</td>
      <td>0.4926</td>
      <td>-0.6597</td>
      <td>0.0000</td>
      <td>0.4767</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.8271</td>
      <td>0.6124</td>
      <td>0.3818</td>
      <td>0.3612</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.6369</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.3612</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0000</td>
      <td>0.4559</td>
      <td>-0.2960</td>
      <td>0.0772</td>
      <td>0.5574</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.6124</td>
      <td>0.0000</td>
      <td>0.0772</td>
      <td>0.6808</td>
      <td>-0.3182</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.3182</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5994</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4215</td>
      <td>-0.8481</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.5719</td>
      <td>0.0000</td>
      <td>0.1531</td>
      <td>0.7717</td>
      <td>0.2584</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.1027</td>
      <td>0.5826</td>
      <td>0.4215</td>
      <td>0.3182</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0000</td>
      <td>0.6825</td>
      <td>0.0000</td>
      <td>-0.3400</td>
      <td>0.5106</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.2732</td>
      <td>0.1139</td>
      <td>0.6470</td>
      <td>0.3612</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0000</td>
      <td>0.6514</td>
      <td>-0.1280</td>
      <td>0.5267</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5719</td>
      <td>0.0000</td>
      <td>0.6486</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.4588</td>
      <td>0.0000</td>
      <td>-0.5267</td>
      <td>0.0000</td>
      <td>0.1298</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0000</td>
      <td>0.7067</td>
      <td>0.4215</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.3400</td>
      <td>0.0000</td>
      <td>0.2716</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-0.6486</td>
      <td>0.6988</td>
      <td>-0.2960</td>
      <td>0.3612</td>
      <td>-0.5423</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.4404</td>
      <td>0.0000</td>
      <td>0.5096</td>
      <td>0.0000</td>
      <td>0.8555</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.3818</td>
      <td>0.8979</td>
      <td>0.6581</td>
      <td>0.3400</td>
      <td>-0.1007</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.4215</td>
      <td>-0.7506</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.6369</td>
      <td>0.7232</td>
      <td>0.0000</td>
      <td>-0.8398</td>
      <td>-0.3382</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.7845</td>
      <td>0.8513</td>
      <td>0.0000</td>
      <td>-0.4939</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.0000</td>
      <td>-0.2263</td>
      <td>0.5267</td>
      <td>0.0772</td>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.3595</td>
      <td>0.8591</td>
      <td>0.0000</td>
      <td>-0.4588</td>
      <td>-0.3597</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4019</td>
      <td>0.3818</td>
      <td>-0.5267</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.4588</td>
      <td>0.0000</td>
      <td>0.5983</td>
      <td>0.5267</td>
      <td>-0.7430</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.5279</td>
      <td>0.2500</td>
      <td>0.0772</td>
      <td>0.6369</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.2500</td>
      <td>0.0000</td>
      <td>0.2732</td>
      <td>-0.4019</td>
      <td>0.3400</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.0000</td>
      <td>0.5719</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5574</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.8765</td>
      <td>0.8016</td>
      <td>0.0000</td>
      <td>0.7430</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-0.2263</td>
      <td>0.8689</td>
      <td>0.4019</td>
      <td>0.5859</td>
      <td>-0.8316</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.5719</td>
      <td>0.7906</td>
      <td>0.0000</td>
      <td>0.5267</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.5574</td>
      <td>0.5994</td>
      <td>0.2023</td>
      <td>0.3612</td>
      <td>0.6467</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.6249</td>
      <td>0.9466</td>
      <td>0.0000</td>
      <td>0.5267</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>83</th>
      <td>-0.1803</td>
      <td>-0.5655</td>
      <td>0.0000</td>
      <td>-0.1531</td>
      <td>0.1280</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.7096</td>
      <td>0.5213</td>
      <td>-0.3400</td>
      <td>0.3612</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.5106</td>
      <td>0.9217</td>
      <td>0.0000</td>
      <td>0.6808</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.0000</td>
      <td>0.3612</td>
      <td>0.2500</td>
      <td>0.0000</td>
      <td>0.1027</td>
    </tr>
    <tr>
      <th>87</th>
      <td>-0.5267</td>
      <td>0.0000</td>
      <td>-0.4939</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.0000</td>
      <td>0.5106</td>
      <td>0.0000</td>
      <td>-0.4754</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.9169</td>
      <td>0.4199</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1779</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.4019</td>
      <td>0.1511</td>
      <td>0.0772</td>
      <td>0.0000</td>
      <td>0.6369</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.0000</td>
      <td>0.3182</td>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>-0.1531</td>
    </tr>
    <tr>
      <th>92</th>
      <td>-0.8360</td>
      <td>0.5423</td>
      <td>0.5267</td>
      <td>0.0000</td>
      <td>-0.6195</td>
    </tr>
    <tr>
      <th>93</th>
      <td>-0.5574</td>
      <td>0.7777</td>
      <td>0.2732</td>
      <td>-0.4588</td>
      <td>-0.4019</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.0000</td>
      <td>0.8126</td>
      <td>0.5719</td>
      <td>0.3612</td>
      <td>0.1779</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.0000</td>
      <td>0.6476</td>
      <td>0.6369</td>
      <td>-0.1531</td>
      <td>-0.5267</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.0000</td>
      <td>0.5754</td>
      <td>0.0000</td>
      <td>-0.4754</td>
      <td>-0.6705</td>
    </tr>
    <tr>
      <th>97</th>
      <td>-0.3818</td>
      <td>0.0000</td>
      <td>-0.6124</td>
      <td>0.5994</td>
      <td>0.7184</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.4391</td>
      <td>0.0772</td>
      <td>-0.2960</td>
      <td>0.0000</td>
      <td>0.4215</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.0000</td>
      <td>-0.2263</td>
      <td>0.4019</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.6915</td>
      <td>-0.2263</td>
      <td>0.4019</td>
      <td>0.3182</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>




```python
x = np.arange(100)

plt.scatter(x,pivot_df['@BBC'],marker='o')
plt.scatter(x,pivot_df['@CBS'],marker='o')
plt.scatter(x,pivot_df['@CNN'],marker='o')
plt.scatter(x,pivot_df['@FoxNews'],marker='o')
plt.scatter(x,pivot_df['@nytimes'],marker='o')

#Plot labels
plt.legend(loc='best',bbox_to_anchor=(1,1))
plt.title("Sentiment Analysis of Media Tweets (12/10/17)")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid=True
plt.xlim([100,0])
plt.ylim([-1,1])

#Save Plot
plt.savefig("Tweet.png")

plt.show()

```


![png](output_8_0.png)



```python
group_df = sentiment_df.groupby('Data_Source').mean()
group_df = group_df['compound']
group_df
```




    Data_Source
    @BBC        0.170004
    @CBS        0.374197
    @CNN        0.020133
    @FoxNews    0.072199
    @nytimes   -0.007443
    Name: compound, dtype: float64




```python

plt.bar(1,group_df['@BBC'])
plt.bar(2,group_df['@CBS'])
plt.bar(3,group_df['@CNN'])
plt.bar(4,group_df['@FoxNews'])
plt.bar(5,group_df['@nytimes'])


plt.title("Overall Sentiment Analysis of Media Tweets (12/10/17)")
plt.ylabel("Tweet Polarity")
plt.xticks(np.arange(5), ('BBC', 'CBS', 'CNN', 'Fox News', 'NY Times') )
plt.ylim(-.1,.5)
plt.savefig("Overall.png")
plt.show()
```


![png](output_10_0.png)

