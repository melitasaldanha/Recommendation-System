# README

## Objective:
Your objective is to perform **item-item collaborative filtering** over the provided products and ratings.

## Datasets:
1. [Software_5.json.gz](https://www.google.com/url?q=http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software_5.json.gz&sa=D&ust=1601864790142000&usg=AOvVaw0FnX86ruuuzbtSx2UYstId)
(N = 12,805 Reviews)

2. [Books_5.json.gz](https://www.google.com/url?q=http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books_5.json.gz&sa=D&ust=1601864790143000&usg=AOvVaw34Md3R1kb9K5uSdvQ21Ja4)
(N = 27,164,983 Reviews)

## Details:

To prepare the system, you will first need to do some filtering:
1. Filter to only one rating per user per item by taking their most recent rating (or their last within the data file; as long as you have one rating per person it is fine)
2. From there, filter to items associated with at least 25 distinct users 
3. From there, filter to users associated with at least 5 distinct items

_Option:_ If you have a particular RDD that has less than an order of 1k entries (i.e. a list of reviewerIDs or asins), at that point, it's ok to collect them into a sc.Broadcast variable.

Then, apply item-item collaborative filtering to predict missing values for the rows prescribed in the output. Use the following settings for your collaborative filtering:
1. Use 50 neighbors (or all possible neighbors if < 50 have values) with the weighted average approach (weighted by similarity). Do not include neighbors:
    - with negative or zero similarity or
    - those having less than 2 columns (i.e. users) with ratings for whom the target row (i.e. the intersection of users_with_ratings for the two is < 2; can check for this before checking similarity).
2. Within a target row, do not make predictions for columns (i.e. users) that do not have at least 2 neighbors with values 
3. Only need to focus on the specified items (in practice, you wouldn't store a completed utility matrix, rather this represents querying the recommendation system, given an item, for users that might be interested in the item).
