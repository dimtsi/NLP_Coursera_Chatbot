import os
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        print(self.thread_embeddings_folder)
        print(tag_name)
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        #### YOUR CODE HERE ####
        question_vec = question_to_vec(question, self.word_embeddings, dim=self.embeddings_dim)
        #### YOUR CODE HERE ####
        # best_thread = pairwise_distances_argmin(question_vec.reshape(1, -1), thread_embeddings, metric='cosine')[0]
        scores_list = []
        k = 10
        n = int(len(thread_ids) / k)
        for i in range(k):
            print(i)
            if i == k - 1:
                best_thread, dist = pairwise_distances_argmin_min(question_vec.reshape((1, self.embeddings_dim)),
                                                                  thread_embeddings[i * n:, :], metric='cosine')
            else:
                best_thread, dist = pairwise_distances_argmin_min(question_vec.reshape((1, self.embeddings_dim)),
                                                                  thread_embeddings[i * n:(i + 1) * n, :],
                                                                  metric='cosine')

            scores_list.append({'thread': i * n + best_thread[0], 'dist': dist[0]})

        df = pd.DataFrame(scores_list).sort_values(by='dist')
        best_thread = int(df.iloc[0]['thread'])
        return thread_ids[best_thread]

# class DialogueManager(object):
#     def __init__(self, paths):
#         print("Loading resources...")
#         # Create chitchatbot
#         self.chitchatbot = self.create_chitchat_bot()
#         # Intent recognition:
#         self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
#         self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])
#
#         self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'
#
#         # Goal-oriented part:
#         self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
#         self.thread_ranker = ThreadRanker(paths)
#
#     def create_chitchat_bot(self):
#         """Initializes self.chitchat_bot with some conversational model."""
#
#         # Hint: you might want to create and train chatterbot.ChatBot here.
#         # It could be done by creating ChatBot with the *trainer* parameter equals
#         # "chatterbot.trainers.ChatterBotCorpusTrainer"
#         # and then calling *train* function with "chatterbot.corpus.english" param
#
#         ########################
#         #### YOUR CODE HERE ####
#         ########################
#         chatbot = ChatBot('dimtsi chitchatbot')
#         trainer = ChatterBotCorpusTrainer(chatbot, show_training_progress=False)
#         trainer.train("chatterbot.corpus.english")
#         return chatbot
#
#     def generate_answer(self, question):
#         """Combines stackoverflow and chitchat parts using intent recognition."""
#
#         # Recognize intent of the question using `intent_recognizer`.
#         # Don't forget to prepare question and calculate features for the question.
#
#         #### YOUR CODE HERE ####
#         prepared_question = text_prepare(question)
#         #### YOUR CODE HERE ####
#         features = self.tfidf_vectorizer.transform([prepared_question])  # needs list
#         print(features.shape)
#         #### YOUR CODE HERE ####
#         intent = self.intent_recognizer.predict(features)
#
#         # Chit-chat part:
#         if intent == 'dialogue':
#             # Pass question to chitchat_bot to generate a response.
#             #### YOUR CODE HERE ####
#             response = self.chitchatbot.get_response(question)
#             return response
#
#         # Goal-oriented part:
#         else:
#             # Pass features to tag_classifier to get predictions.
#             #### YOUR CODE HERE ####
#             tag = self.tag_classifier.predict(features)[0]
#             print("tag: {}".format(tag))
#
#             # Pass prepared_question to thread_ranker to get predictions.
#             #### YOUR CODE HERE ####
#             thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
#             return self.ANSWER_TEMPLATE % (tag, thread_id)


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")
        # Create chitchatbot
        self.chitchatbot = self.create_chitchat_bot()
        # Intent recognition:
        self.paths = paths

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param

        ########################
        #### YOUR CODE HERE ####
        ########################
        chatbot = ChatBot('dimtsi chitchatbot')
        trainer = ChatterBotCorpusTrainer(chatbot, show_training_progress=False)
        trainer.train("chatterbot.corpus.english")
        return chatbot

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.

        #### YOUR CODE HERE ####
        intent_recognizer = unpickle_file(self.paths['INTENT_RECOGNIZER'])
        tfidf_vectorizer = unpickle_file(self.paths['TFIDF_VECTORIZER'])
        prepared_question = text_prepare(question)
        #### YOUR CODE HERE ####
        features = tfidf_vectorizer.transform([prepared_question])  # needs list
        print(features.shape)
        #### YOUR CODE HERE ####
        intent = intent_recognizer.predict(features)

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            #### YOUR CODE HERE ####
            response = self.chitchatbot.get_response(question)
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            #### YOUR CODE HERE ####
            tag = self.tag_classifier.predict(features)[0]
            print("tag: {}".format(tag))

            # Pass prepared_question to thread_ranker to get predictions.
            #### YOUR CODE HERE ####
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
            return self.ANSWER_TEMPLATE % (tag, thread_id)


