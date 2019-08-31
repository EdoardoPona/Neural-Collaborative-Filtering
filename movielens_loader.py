
class MovielensDatasetLoader:
    def __init__(self, filename='ml-1m/ratings.dat'):
        self.ratings = open(filename, 'r').readlines()
        self.user_num, self.item_num = 6040, 3952        # as specified by the README
        # strangely only user corresponds to the length of the dictionary some items are missing

    def build_dictionaries(self):
        ratings = open('ml-1m/ratings.dat', 'r').readlines()

        user_to_movie = {}
        movie_to_user = {}

        for rating in ratings:
            user_id, movie_id, rating, _ = rating.split('::')
            rating = int(rating)

            if user_id in user_to_movie.keys():
                # this assumes there aren't multiple ratings for a user-book pair, which there shouldn't
                # might be better to check first though
                user_to_movie[user_id][movie_id] = rating
            else:
                user_to_movie[user_id] = {movie_id: rating}

            if movie_id in movie_to_user.keys():
                movie_to_user[movie_id][user_id] = rating
            else:
                movie_to_user[movie_id] = {user_id: rating}

        user_num = len(user_to_movie.keys())
        movie_num = len(movie_to_user.keys())

        print(user_num, 'users', movie_num, 'movies')
        return user_to_movie, movie_to_user

user_num, item_num = 6040, 3952           # as specified by the README   strangely only uesr_num corresponds to the length of the dictionary
                                    # some items are missing
