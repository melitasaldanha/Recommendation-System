from __future__ import print_function
from pyspark.sql import SparkSession 
from pyspark.sql import SQLContext
from scipy.spatial.distance import cosine
from operator import add

import sys
import re
import numpy as np
import math


# Get:  For key = (user, item), get 2 values: (time1, rating1), (time2, rating2)
# Return: (time, rating) pair with highest time, out of the 2 pairs received as input
def reduce_user_item_pair(a, b):
    d1, r1 = a
    d2, r2 = b

    if(d1>d2):
      return a
    else:
      return b

# Get (key, list of (inner_key, inner_val))
# Return (inner_key, list of (key, inner_val))
def rearrange_pairs(x):
  k, v = x;
  l = []
  for val in v:
    ik, iv = val
    l.append((ik, [(k, iv)]))
  return l


# If a row's missing ratings are not to be predicted,
# Find the users without a rating, and print rating as 0
def missing_user_rating_zero(missing_user_set):
  new_dict = {}
  for user in missing_user_set:
    new_dict[user] = 0
  return new_dict


# Normalize the row of the utility matrix
# Do this by subtracting its mean from each of the ratings 
# Return the row in the same format with new rating values
def find_mean(item, user_rating):
  mean = np.mean(list(user_rating.values()))

  new_dict = {}
  for (user, rating) in user_rating.items():
    new_dict[user] = rating-mean

  return (item, new_dict)


# Find all user's whose rating is not present for a given item
# Make the rating of such users as 0
# Sort the (user, rating) map based on user (so that rows of all items have ratings of user in same order)
# Return (item, numpy_array(ratings)) with ratings in sorted order of users
def find_ratings_numpy_array(x):
  item, user_rating = x

  for user in broadcast_user_set.value:
    if user not in user_rating.keys():
      user_rating[user] = 0

  user_rating = sorted(user_rating.items(), key=lambda a: a[0])
  ratings = np.array([r[1] for r in user_rating])
  return (item, ratings)


# Given: (neighbour_item, user_rating_map_for_neighbor_item, user_rating_map_for_queried_items)
def find_similarity(x, user_rating_map, input_item_row):

  # Returns: (item, new_dict_user_rating_map_after_mean)
  x = find_mean(x, user_rating_map)
  
  # Returns: (item, numpy_array_of_ratings)
  x = find_ratings_numpy_array(x)

  # Find the cosine similarity for the neighbor_item based on the given numpy array of queried_item and neighbor_item
  # Return: (neighbour_item, similarity_with_queried_item)
  return (x[0], cosine(input_item_row, x[1]))


# Predict rating for user whose rating is not present
# Given: (user_to_be_predicted, neighbours_rdd), where format of neighbor rdd: (neighbor_item, similarity_with_queried_item)
# Apply formula for collaborative filtering
# Return: predicted rating
def predict_rating(user, neighbours_rdd):

  hm = broadcast_item_user_rating_map.value
  
  sum_num, sum_den = 0, 0
  for (neighbor, sim) in neighbours_rdd.items():
    try:
      sum_num += (sim * hm[neighbor][user])
      sum_den += sim
    except KeyError:
      continue

  if(sum_den!=0):
    return sum_num/sum_den
  else:
    return 0


def query(item_id):

  # Get map(item, map(user, rating)) from broadcast variable
  hm = broadcast_item_user_rating_map.value
  user_rating_map = hm[item_id]

  # Find set of all users whose rating is present for queried item
  user_set = set(user_rating_map.keys())

  # From universal set of user, find set of remaining users whose rating is not present
  missing_user_set = (broadcast_user_set.value)-user_set

  # Find numpy array of ratings for queried_item
  ratings_numpy_array = find_ratings_numpy_array(find_mean(item_id, user_rating_map))

  # If less than 2 users have ratings present, do not predict rating for remaining users
  # For the remaining users print rating 0 and return map (user, rating)
  if(len(user_rating_map)<2):
    user_rating_map.update(missing_user_rating_zero(missing_user_set))
    return user_rating_map

  # Aim: Find all neighbor items of queried item
  # Step 1: Find universal set of items from broadcast variable
  # Step 2: Remove queried_item from this list
  # Step 3: Filter items whose intersection of users with queried_items users is less than 2
  # Step 4: Find similarity of each item with queried_item: Format = (neighbor_item, similarity)
  # Step 5: Remove neighbor_item whose similarity <= 0 or is nan
  # Step 6: Sort neighbors in decresing order of similarity
  # Step 7: Take top 50 neighbors or total number of neighbors (whichever is less)
  neighbours_rdd = spark.sparkContext.parallelize(broadcast_items_list.value) \
                                    .filter(lambda x: x!=item_id and len(user_set.intersection(set(hm[x].keys())))>=2) \
                                    .map(lambda x: find_similarity(x, hm[x], ratings_numpy_array[1])) \
                                    .filter(lambda x: x[1]>0 and not(math.isnan(x[1]))) \
                                    .sortBy((lambda x: x[1]), False) \
                                    .take(50)

  # If number of neighbors left after filtering < 2, do not predict ratings for users whose rating is not present
  # For such users, print rating = 0 and return map (user, rating)
  if(len(neighbours_rdd) < 2):
    user_rating_map.update(missing_user_rating_zero(missing_user_set))
    return user_rating_map

  # Make list of neighbors to a map
  neighbours_rdd = dict(neighbours_rdd)

  # Predict ratings for users whose rating is not present and save as map (user, rating)
  missing_user_ratings_predicted = spark.sparkContext.parallelize(missing_user_set) \
                                            .map(lambda x: (x, predict_rating(x, neighbours_rdd))) \
                                            .collectAsMap()

  # Return map of (user, rating) as result
  user_rating_map.update(missing_user_ratings_predicted) 
  return user_rating_map


if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("Hypothesis Testing")\
        .getOrCreate()

    sqlContext = SQLContext(spark.sparkContext)
    
    # Find each line of json with given keys
    lines = sqlContext.read.json(sys.argv[1]).select("reviewerName", "asin", "unixReviewTime", "overall")
    
    # Create a map of ((user, item), (time, rating))
    # For each key (user, item), return the value with highest time
    # So each (user, item) pair in final rdd is unique
    user_item_pair = lines.rdd.map(lambda row: row) \
                              .map(lambda x: ((x[0], x[1]), (x[2], x[3]))) \
                              .reduceByKey(lambda a,b: reduce_user_item_pair(a, b))
                      
    # Change format of previous rdd to: (item, (user, rating))
    # Reduce to make: (item, list of (user, rating) pairs for this key item)
    # If list of values for key item contains less than 25 (user, rating) pairs, remove item
    count_users_per_item = user_item_pair.map(lambda x: (x[0][1], [(x[0][0], x[1][1])])) \
                                        .reduceByKey(add) \
                                        .filter(lambda x: x[0]!=None and len(x[1])>=25)

    # Change format of previous rdd to: (user, (item, rating))
    # Reduce to make: (user, list of (item, rating) pairs for this key user)
    # If list of values for key user contains less than 5 (item, rating) pairs, remove user
    count_items_per_user = count_users_per_item.map(lambda x: rearrange_pairs(x)) \
                                                .flatMap(lambda x: x) \
                                                .reduceByKey(add) \
                                                .filter(lambda x: x[0]!=None and len(x[1])>=5)

    # Get universal set of users left after filtering
    get_users = set(count_items_per_user.map(lambda x: x[0]).collect())
    broadcast_user_set = spark.sparkContext.broadcast(get_users)
                                              
    # Change format to: (item, map of (user, rating))
    final_list = count_items_per_user.map(lambda x: rearrange_pairs(x)) \
                                      .flatMap(lambda x: x) \
                                      .reduceByKey(add) \
                                      .map(lambda x: (x[0], dict(x[1])))

    # Get all items
    get_items = final_list.map(lambda x: x[0]).collect()
    broadcast_items_list = spark.sparkContext.broadcast(get_items)

    # Save map of final_list rdd as a broadcast variable to have readonly access for each query
    # Format of map: key=item, value=(map of (user, rating))
    broadcast_map_value = final_list.collectAsMap()
    broadcast_item_user_rating_map = spark.sparkContext.broadcast(broadcast_map_value)

    # Items to be queried
    input_list = eval(sys.argv[2])
    
    # Find recommendation for each item given as input
    for item_id in input_list:

      print("Item: ", item_id)

      # Query an item using item-item collaborative filtering
      d = query(item_id)

      # Print results of query: Map 
      # Format of map: Key = reviewer_id, value = given rating / predicted rating (if not present originally)
      for i in d.items():
        print(i)
      print('\n')

    spark.stop()