import pickle
import Price_Prediction

with open("rf.pkl", "rb") as f:
    model = pickle.load(f)

neighbourhood_group = input("Enter Neighborhood Group: Manhattan - 1, Brooklyn - 2, Queens - 3, Bronx - 4, Staten Island - 5  >> ")
room_type = input("Enter Room Type: Entire House - 1, Private room - 2, Shared room - 3, Hotel room - 4 >> ")
number_of_reviews = int(input("Enter number_of_reviews: "))
calculated_host_listings_count = int(input("Enter calculated_host_listings_count: "))

d = [[neighbourhood_group, room_type, number_of_reviews, calculated_host_listings_count]]
# d = [[1, 2, 10, 80]]

res = model.predict(d)
print("Result: ", res)



