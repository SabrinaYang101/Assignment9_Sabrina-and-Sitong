# Sabrina Yang, Sitong Xu
import math, os, re
from typing import Tuple, List, Dict
import chardet

class Document:
    """The Document class.
        Attributes: text - the text of the document
                    terms - a dictionary mapping words to the number of times they occur in the document
                         - please note that this sort of dictionary is returned by tokenize
                    term_vector - an ordered list of tfidf values for each term. The length of this list
                                  will be the same for all documents in a corpus.
    """

    def __init__(self):
        """Creates an empty document.
        """
        self.text = ""
        self.terms = {}
        self.term_vector = []

    def __str__(self):
        """Returns the first 500 characters in the document as a preview.
        """
        return self.text[:500]


class TFIDF_Engine:
    """The TFIDF_Engine class.
        Attributes: corpus_location - a relative path to the folder of documents in the corpus
                    documents - a list of document objects, initially empty
                    N - the number of documents in the corpus
                    df_table - a dictionary mapping words to the number of documents they occurred in
                    term_vector_words - an ordered list of the unique words in the corpus. This list
                                will dictate the order of the scores in each document.term_vector
    """

    def __init__(self):
        self.corpus_location = "news_corpus"
        self.documents = []
        self.N = 0
        self.df_table = {}
        self.term_vector_words = []

    def __str__(self):
        s = "corpus has: " + str(self.N) + " documents\n"
        s += "beginning of doc vector words is: \n" + str(self.term_vector_words[:25])
        return s


    def read_files(self):
        try:
            # Get all files in the directory
            files = next(os.walk(self.corpus_location))[2]

            # Filter out hidden files and optionally check for a specific file extension (e.g., .txt)
            files = [file for file in files if not file.startswith('.') and file.endswith('.txt')]

            # Debugging: Print the number of files found after filtering
            print(f"Number of files after filtering: {len(files)}")

            if len(files) != 122:
                print("Warning: The number of documents does not match the expected count.")

            for file in files:
                path = os.path.join(self.corpus_location, file)
                try:
                    with open(path, 'rb') as f:
                        raw_data = f.read()
                        result = chardet.detect(raw_data)
                        encoding = result['encoding']

                    with open(path, 'r', encoding=encoding, errors='ignore') as f:
                        text = f.read()

                    doc = Document()
                    doc.text = text
                    doc.terms = self.tokenize(text)
                    self.documents.append(doc)
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

            self.N = len(self.documents)

        except Exception as e:
            print(f"Error setting up the file reading process: {e}")

    def create_df_table(self):
        for doc in self.documents:
            for term in doc.terms:
                if term in self.df_table:
                    self.df_table[term] += 1
                else:
                    self.df_table[term] = 1
        self.term_vector_words = list(self.df_table.keys())

    def create_term_vector(self, d: Document):
        d.term_vector = []
        for term in self.term_vector_words:
            if term in d.terms:
                tf = 1 + math.log(d.terms[term])
                idf = math.log(self.N / self.df_table[term])
                d.term_vector.append(tf * idf)
            else:
                d.term_vector.append(0)

    def create_term_vectors(self):
        for doc in self.documents:
            self.create_term_vector(doc)

    def calculate_cosine_sim(self, d1: Document, d2: Document) -> float:
        dot_product = sum(a * b for a, b in zip(d1.term_vector, d2.term_vector))
        magnitude_d1 = math.sqrt(sum(a * a for a in d1.term_vector))
        magnitude_d2 = math.sqrt(sum(b * b for b in d2.term_vector))
        if magnitude_d1 == 0 or magnitude_d2 == 0:
            return 0
        return dot_product / (magnitude_d1 * magnitude_d2)

    def get_results(self, query: str):
        query_doc = Document()
        query_doc.text = query
        query_doc.terms = self.tokenize(query)
        self.create_term_vector(query_doc)

        results = []
        for idx, doc in enumerate(self.documents):
            similarity = self.calculate_cosine_sim(query_doc, doc)
            results.append((similarity, idx))

        # Sorting the results by similarity score in descending order
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def query_loop(self):
        """Asks the user for a query. Utilizes self.get_results. Prints the top 5 results.
        """

        print("Welcome!\n")
        while True:
            try:
                print()
                query = input("Query: ")
                sim_scores = self.get_results(query)
                # display the top 5 results
                print("RESULTS\n")
                for i in range(5):
                    print("\nresult number " + str(i) + " has score " + str(sim_scores[i][0]))
                    print(self.documents[sim_scores[i][1]])

            except (KeyboardInterrupt, EOFError):
                break

        print("\nSo long!\n")

    def tokenize(self, text: str) -> Dict[str, int]:
        """Splits given text into a list of the individual tokens and counts them

        Args:
            text - text to tokenize

        Returns:
            a dictionary mapping tokens from the input text to the number of times
            they occurred
        """
        tokens = []
        token = ""
        for c in text:
            if (
                    re.match("[a-zA-Z0-9]", str(c)) != None
                    or c == "'"
                    or c == "_"
                    or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())

        # make a dictionary mapping tokens to counts
        d_tokens = {}
        for t in tokens:
            if t in d_tokens:
                d_tokens[t] += 1
            else:
                d_tokens[t] = 1

        return d_tokens


# Optional Challenge:
# Different to previous assignments, the test cases contained here will likely not show up
# on pycharm where you can directly see how many of them passes or fails.
# Based on the pytest documentation and your experience with previous assignments, see
# if you can create a tfidf_test.py by converting the test cases below to the pytest format.
# Pytest documentation : https://docs.pytest.org/en/7.1.x/getting-started.html

if __name__ == "__main__":
    t = TFIDF_Engine()

    # read the files , populating self.documents and self.N
    t.read_files()

    assert t.N == 122, "read files N test"
    assert t.documents[5].text != "", "read files document text test"
    assert t.documents[100].terms != {}, "read files document terms test"
    assert isinstance(t.documents[9].terms["the"], int), "read files document terms structure test"

    # create self.df_table from the documents
    t.create_df_table()

    assert t.df_table["the"] == 122, "df_table 'the' count test"
    assert t.df_table["star"] == 102, "df_table 'star' count test"
    assert 11349 <= len(t.df_table) <= 11352, "df_table number of unique words test"

    # # create the document vector for each document
    t.create_term_vectors()
    #
    assert len(t.documents[10].term_vector) == len(t.term_vector_words), "create_term_vectors test"

    # # tests for calculate_cosine_sim
    assert t.calculate_cosine_sim(t.documents[0], t.documents[1]) > 0, "calculate_cosine_sim test 1"
    assert t.calculate_cosine_sim(t.documents[0], t.documents[1]) < 1, "calculate_cosine_sim test 1"
    assert abs(t.calculate_cosine_sim(t.documents[0], t.documents[0]) - 1) < 0.01

    # # tests for get_results
    assert t.get_results("star wars")[0][1] == 111, "get_results test 1"
    assert "Lucas announces new 'Star Wars' title" in t.documents[
        t.get_results("star wars")[0][1]].text, "get_results test 1"
    assert t.get_results("movie trek george lucas")[2][1] == 24, "get_results test 2"
    assert "Stars of 'X-Men' film are hyped, happy, as comic heroes" in t.documents[
        t.get_results("movie trek george lucas")[2][1]].text
    assert len(t.get_results("star trek")) == len(t.documents), "get_results test 3"
    #
    # t.query_loop() #uncomment this line to try out the search engine
    #
    #
