
import numpy as np
import time

class GroupsMatcher:
    A, F, N, v, S, V, X, x= 0, 1, 2, 3, 4, 5, 6, 7
    int_to_str= {
        0: ['A', 'Affri'], # A = Affricatives
        1: ['F', 'Fri'],   # F = Fricatives
        2: ['N', 'Nasa'],  # N = Nasals
        3: ['v', 'SemVo'], # v = Semi-vowels
        4: ['S', 'Stop'],  # S = Stops
        5: ['V', 'Vow'],   # V = Vowels
        6: ['X', 'SIL'],   # X = SIL
        7: ['x', 'SIL-']   # x = SIL- (penalty silence)
    }
    groups= [int_to_str[i][0] for i in int_to_str]

    # scores[i]= scores of changing a i into ...
    scores= dict()
    scores['A']= {'A': 5, 'F': 2, 'N': -1, 'v': -1, 'S': 1, 'V': -50, 'X':  0, 'x': 0}
    scores['F']= {'A': 2, 'F': 5, 'N': -1, 'v': -1, 'S': 1, 'V': -50, 'X':  0, 'x': 0}
    scores['N']= {'A': -1, 'F': -1, 'N': 5, 'v': -1, 'S': 0, 'V': -50, 'X':  0, 'x': 0}
    scores['v']= {'A': -1, 'F': -1, 'N': -1, 'v': 5, 'S': -1, 'V': -5, 'X':  0, 'x': 0}
    scores['S']= {'A': -1, 'F': -1, 'N': -1, 'v': -1, 'S': 5, 'V': -50, 'X':  0, 'x': 0}
    scores['V']= {'A': -1, 'F': -5, 'N': -1, 'v': -1, 'S': -1, 'V': 5, 'X':  0, 'x': 0}
    scores['X']= {'A': -1, 'F': -1, 'N': -1, 'v': -1, 'S': -1, 'V': -50, 'X': 5, 'x': 3}

    # silences[i]= possiblity of having a penalty silence between a i and next group ...
    silences= dict()
    silences['A']= {'A': 1, 'F': 0, 'N': 0, 'v': 0, 'S': 1, 'V': 0}
    silences['F']= {'A': 1, 'F': 0, 'N': 0, 'v': 0, 'S': 1, 'V': 0}
    silences['N']= {'A': 1, 'F': 0, 'N': 0, 'v': 0, 'S': 1, 'V': 0}
    silences['v']= {'A': 1, 'F': 0, 'N': 0, 'v': 0, 'S': 1, 'V': 0}
    silences['S']= {'A': 1, 'F': 0, 'N': 0, 'v': 0, 'S': 1, 'V': 0}
    silences['V']= {'A': 1, 'F': 0, 'N': 0, 'v': 0, 'S': 1, 'V': 0}

    # create scoring matrix
    scoring= np.array([[scores[group][group_k] for group_k in groups] for group in groups[:-1]])
    scoring= np.transpose(scoring)


    # create silences matrix
    silences= np.array([[silences[group][group_k] for group_k in groups[:-2]] for group in groups[:-2]])


    # factor for same group move. (horizontal move). =0 : no effect.
    same_group_malus_factor= 1

    silence_tagging_malus= -100
    silence_delay_malus= -2

    def __init__(self, target, concat_consec= False, VERBOSE=0):
        # Concatenate vowels for matching boolean. Concatenates if True
        self.concat_consec= concat_consec
        self.logger= Logger(VERBOSE)

        # Preprocess target sequence
        starttime= time.time()
        self.target= self.preprocess_sentence(target)
        self.logger.time('Automaton Sentence preprocessing', time.time()-starttime)

        self.n= self.target.size

    # PREPROCESS TARGET
    def preprocess_sentence(self, sentence):
        target= []
        for word in sentence:
            target+= [self.X]
            word= np.array([eval('self.' + ele) for ele in word], dtype=np.int8)

            # Concatenate vowels together so that 2 consecutives vowels are found as one
            if self.concat_consec:
                to_del= []
                for i in range(len(word)-1):
                    if word[i]!= 2: # if the character is not a vowel
                        continue
                    if word[i]==word[i+1]: # if they are consecutives
                        to_del.append(i)
                word= np.delete(word, to_del)

            # Insert spaces between group to let the automat the ability to create spaces
            new_word= []
            for i in range(len(word) - 1):
                new_word.append(word[i])
                if self.silences[word[i]][word[i+1]]:
                    new_word.append(self.x)
            new_word.append(word[-1])

            target+= new_word

        target+= [self.X]

        return np.array(target)

    # SAMPLE IS FOR RF OUTPUT, ONE PREDICTION PER TIME FRAME.
    def define_sample(self, sample):
        self.sample= np.array([eval('self.' + ele) for ele in sample], dtype=np.int8)
        self.m= self.sample.size
        self.compute_score_matrix_scores() # Compute the score array

    def define_sample_with_VAD(self, sample, active_voice_segments):
        self.sample= np.array([eval('self.' + ele) for ele in sample], dtype=np.int8)
        self.m= self.sample.size

        starttime= time.time()
        self.compute_score_matrix_scores_with_VAD(active_voice_segments) # Compute the score array
        self.logger.time('Automaton Compute score matrix', time.time()-starttime)

    # PROBAS IS FOR BLSTM OUTPUT, 7 PROBA PER TIME FRAME.
    def define_probas(self, probas):
        self.m= probas.shape[0]
        self.compute_score_matrix_probas(probas)
        # assert self.m > self.n, "sample must be longer than target"

    def find_best_match(self):
        # Find the best match from the sample sequence according to the target sequence

        starttime= time.time()
        self.compute_matrix() # Compute the array that represents the best path
        self.logger.time('Automaton Compute path matrix', time.time()-starttime)


        starttime= time.time()
        bestmatch= self.traceback() # Find the best path by doing traceback through the array
        self.logger.time('Automaton Traceback', time.time()-starttime)

        return self.intlist_to_charlist(bestmatch)

    def compute_score_matrix_scores(self):
        self.S= np.zeros((self.n, self.m), dtype=np.int16)
        for j in range(self.m):
            for i in range(self.n):
                self.S[i,j]= self.scoring[self.target[i], self.sample[j]]

    def compute_score_matrix_scores_with_VAD(self, active_voice_segments):
        self.S= np.zeros((self.n, self.m), dtype=np.int16)
        # print self.target
        for j in range(self.m):
            is_active= False
            # print self.sample[j], j,
            for segment in active_voice_segments:
                if (j >= segment[0]) & (j <= segment[1]):
                    # print segment,
                    is_active= True
                    break
            # print is_active,
            st= []
            for i in range(self.n):
                self.S[i,j]= self.scoring[self.target[i], self.sample[j]]
                if not is_active:
                    if not ((self.target[i] == self.X) or (self.target[i] == self.x)):
                        self.S[i,j]+= self.silence_tagging_malus
                else:
                    if (self.target[i] == self.x):
                        self.S[i,j]+= self.silence_delay_malus
                st.append("%s"% (self.S[i,j]))
            # print ' '.join(st)

    def compute_score_matrix_probas(self, probas):
        self.S= np.zeros((self.n, self.m), dtype=np.float32)
        for i in range(self.n):
            if self.target[i] != self.X:
                for j in range(self.m):
                    self.S[i,j]= probas[j, self.sorted_matching[self.target[i]]]

    def compute_matrix(self):
        k= self.same_group_malus_factor

        path_boundaries_S= [0]
        path_boundaries_E= [0]
        step_S= 0
        step_E= 0
        for i in range(1, self.n):
            if self.target[i-1] != self.X:
                step_S+= 1
            if self.target[::-1][i-1] != self.X:
                step_E+= 1
            path_boundaries_S.append(step_S)
            path_boundaries_E.append(step_E)

        path_boundaries_E= [self.m  - ele for ele in path_boundaries_E][::-1]

        # Initialize the matrix with -Inf values.
        self.D= -float('inf')*np.ones((self.n, self.m), dtype=np.int16)
        for i in range(self.n): # fill the array from top to bottom, left to right
            # for j in range(i//2, self.m-(self.n-i-1)//2): # represents all the possible paths
            for j in range(path_boundaries_S[i], path_boundaries_E[i]):
            # for j in range(self.m):
                if (j>0) & (i>0):
                    subjects= [self.D[i-1, j-1] + self.S[i,j]]
                    subjects+= [self.D[i, j-1] + k*self.S[i,j]]
                    if (i>1) & (self.target[i-1] == self.X):
                        subjects+= [self.D[i-2, j-1] + self.S[i,j]]
                    self.D[i,j]= max(subjects)
                else:
                    self.D[i,j]= self.S[i,j]

    def traceback(self):
        self.path= [] # path taken by the automaton (list of tuples)
        alignment= [] # associate corresponding alignment found

        # Final score is the max value between the D[n-2, m-1] and D[n-1, m-1]
        self.score= max(self.D[self.n-2, self.m-1], self.D[self.n-1, self.m-1])

        # We start at the i, j coordinate of the max score cell
        i, j= (self.n-2, self.m-1) if (self.score == self.D[self.n-2, self.m-1]) else (self.n-1, self.m-1)

        while j>0:
            alignment.append(self.target[i])
            self.path.append((i, j))
            j-= 1 # Whatever happens we go left!
            if (i>1):
                subjects= [self.D[i, j], self.D[i-1, j]]
                if self.target[i-1] == self.X: # there is a space between two phonemes, potential jump
                    subjects.append(self.D[i-2, j])
                    if max(subjects) == self.D[i-2, j]: # first check if path is going two lines up
                        i-= 2 # we go 2 lines up
                    elif max(subjects) == self.D[i, j]: # else we check if path is staying on same line
                        pass # we don't change line, we don't do anything
                    else: # else, we go one line up (space)
                        i-= 1 # we go 1 line up
                else:
                    if max(subjects) == self.D[i-1, j]: # first check if path is going one line up
                        i-= 1 # we go 1 line up
                    else:
                        pass # we don't change line, we don't do anything.
            elif (i>0):
                subjects= [self.D[i, j], self.D[i-1, j]]
                if max(subjects) == self.D[i-1, j]: # first check if path is going one line up
                    i-= 1 # we go 1 line up
                else:
                    pass # we don't change line, we don't do anything

        # we add to the alignment the line we are in
        alignment.append(self.target[i])

        # we add the i,j values to the path
        self.path.append((i, j))

        # as the path was going backward we need to reverse it
        alignment.reverse()
        return alignment

    def inter_insert(self, l, item):
        # Insert character item between every elements of the list l
        result= [item] * (len(l) * 2 -1)
        result[0::2] = l
        result= [item] + result + [item]
        return np.array(result)

    def intlist_to_charlist(self, seq):
        return [self.int_to_str[ele][0] for ele in seq]

    def charlist_to_string(self, seq):
        return ' '.join(seq)

    def chain_to_string(self):
        return ' -> '.join([self.int_to_str[i][1] for i in self.target])

if __name__ == "__main__":
    # lists= open('input.txt', 'r').readlines()
    # sample= lists[0][:-1].split()
    # target= lists[1][:-1].split()
    # sample= 'V V V V V N N N N N F F F F F F F F V V V V N V N N N S S S'.split()
    # target= 'v N V S'.split()
    sample= 'V v N N N'.split()
    target= 'V v'.split()


    matcher= GroupsMatcher(target)
    matcher.define_sample(sample)
    bestmatch= matcher.find_best_match()

    print matcher.charlist_to_string(sample)
    print matcher.charlist_to_string(bestmatch)
    print 'score: %s' % (matcher.score)
