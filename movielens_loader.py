import copy

class MovielensDatasetLoader:


    def __init__(self, filename='ml-1m/ratings.dat', mode='user_item'):
        self.ratings = open(filename, 'r').readlines()
        self.user_num, self.item_num = 6040, 3952        # as specified by the README
        # strangely only user corresponds to the length of the dictionary, some items are missing
        self.mode = mode
        assert (mode=='user_item' or mode=='item_item')


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

        print(self.user_num, 'users', self.item_num, 'movies')
        self.user_to_movie = user_to_movie
        self.movie_to_user = movie_to_user

        self._build_test_set()
        return self.get_dictionaries()


    def get_dictionaries(self):
        test_dataset = copy.deepcopy(self.test_user_to_movie) if self.mode=='user_item' else copy.deepcopy(self.test_user_to_movie_pair)
        return copy.deepcopy(self.user_to_movie),\
               copy.deepcopy(self.movie_to_user), \
               test_dataset


    def _build_test_set(self):
        if self.mode == 'user_item':
            test_user_to_movie = {}
            for user in list(self.user_to_movie.keys()):
                # TODO at the moment last_item_key doesn't care about the actual rating
                last_item_key = list(self.user_to_movie[user].keys())[-1]
                test_user_to_movie[user] = last_item_key
                del self.user_to_movie[user][last_item_key]
                del self.movie_to_user[last_item_key][user]
            self.test_user_to_movie = test_user_to_movie

        elif self.mode == 'item_item':
            test_user_to_movie_pair = {}
            for user in list(self.user_to_movie.keys()):
                items = list(self.user_to_movie[user].keys())
                test_user_to_movie_pair[user] = [int(items[-1]), int(items[-2])]
                del self.user_to_movie[user][items[-1]]
                del self.user_to_movie[user][items[-2]]
            self.test_user_to_movie_pair = test_user_to_movie_pair
