import numpy as np
from matplotlib import pyplot as plt
class hmm:
    def __init__(self,num_states, num_obs, states=None, observations=None, A=None, B= None, pie=None):
        """
        :param A: should ne np array num_states X num_states
        :param B: should be np array num_states X num_observations
        :param pi: should be 1 d array
        random intialisation can be done from outside
        """
        self.num_states = num_states
        self.num_obs = num_obs
        if states is None and observations is not None:
            '''intialise random A
            initialise random B
            Initialise random pi
            '''
            self.initialise_uniform()
            pass
        if A is not None:
            self.A = A
            self.B = B
            self.pie = pie
        else:
            self.initialise_special(states, observations)

        # self.num_states = self.A.shape[0]
        # self.num_obs = self.B.shape[1]
        self._name_initialized = False
    def set_state_obs_names(self,state_name_dict, obs_name_dict):
        '''
        :param state_name_dict: dict{state_number : state_name}, state_number is range(0,num_states)
        :param obs_name_dict: dict{obs_number : obs_name}, obs_number is range(0,...,num_obs)
        :return:
        '''
        #validate the dicts here
        self.state_dict = state_name_dict
        self.obs_dict = obs_name_dict
        self._name_initialized = True

    # def initalise_random(self, observations):
    def initialise_uniform(self):
        self.A = np.ones((self.num_states, self.num_states))  # laplacian smoothening
        self.pie = np.ones((self.num_states,))

        denominator = np.sum(self.A, axis=1)
        denominator = denominator.reshape(-1, 1)
        self.A /= denominator
        self.pie /= np.sum(self.pie)

        self.B = np.ones((self.num_states, self.num_obs))  # laplacian kind of smoothening
        denominator = np.sum(self.B, axis=1)
        denominator = denominator.reshape(-1, 1)
        self.B /= denominator

    def initialise_special(self, states, observations):
        if states is None:
            return
        self.A = np.ones((self.num_states, self.num_states))  # laplacian smoothening
        self.pie = np.ones((self.num_states,))
        # self.A *= .005
        for s in states:
            for i in range(s.shape[0] - 1):
                if i == 0:
                    self.pie[s[i]] += 1
                self.A[s[i]][s[i + 1]] += (1)
        denominator = np.sum(self.A, axis=1)
        denominator = denominator.reshape(-1, 1)
        self.A /= denominator
        self.pie /= np.sum(self.pie)

        self.B = np.ones((self.num_states, self.num_obs))  # laplacian kind of smoothening
        # self.B *= .005
        for j in range(len(observations)):
            for i in range(observations[j].shape[0]):
                self.B[states[j][i]][observations[j][i]] += (1)
        denominator = np.sum(self.B, axis=1)
        denominator = denominator.reshape(-1, 1)
        self.B /= denominator

    def initialise(self, states, observations):
        """
        :param states: list of state sequences each a np array
        :param obsverations: list of observation sequences each a np array
        counting the occurrence and normalizing to probability
        """
        # implement initialisation here
        self.A = np.ones((self.num_states, self.num_states)) # laplacian smoothening

        self.pie = np.ones((self.num_states,) )

        for s in states:
            for i in range(s.shape[0]-1):
                if i==0 :
                    self.pie[s[i]] += 1
                self.A[s[i]][s[i+1]] += 1
        denominator = np.sum(self.A, axis =1)
        denominator = denominator.reshape(-1,1)
        self.A /= denominator
        self.pie /= np.sum(self.pie)

        self.B = np.ones((self.num_states, self.num_obs)) # laplacian kind of smoothening

        for j in range(len(observations)):
            for i in range(observations[j].shape[0]):
                self.B[states[j][i]][observations[j][i]] += 1
        denominator = np.sum(self.B, axis =1)
        denominator = denominator.reshape(-1,1)
        self.B /= denominator

    def viterbiAlgorithm(self, obs):
        """
        obs is a 1d array assumed it contains 0,1,2,3,4....starting with 0
        """
        A = self.A
        B = self.B
        pie = self.pie
        best_state = np.zeros(shape=[A.shape[0], obs.shape[0]], dtype=np.int32)
        best_state[:, :] = -1
        best_prob1 = pie * B[:, obs[0]]
        # print("initially best_prob1 is", best_prob1)
        for time in range(1, obs.shape[0]):
            best_prob2 = np.zeros(best_prob1.shape)
            #        print("best_prob1 is",best_prob1)
            for i in range(A.shape[0]):
                p = best_prob1 * A[:, i]
                # print(p.shape)
                best_prob2[i] = np.max(p)
                best_state[i, time] = np.argmax(p)
            best_prob2 = best_prob2 * B[:, obs[time]]
            best_prob1 = best_prob2
            # print("best_prob1 is", best_prob1)
        state_sequence = [np.argmax(best_prob1)]
        i = obs.shape[0] - 1
        while i > 0:
            state_sequence.append(best_state[state_sequence[-1]][i])
            i -= 1
            # print(state_sequence)
        state_sequence.reverse()
        # print('state sequence is',state_sequence)

        return best_state, state_sequence, np.max(best_prob1)

    def viterbiAlgorithm_logscale(self, obs):
        """
        obs is a 1d array assumed it contains 0,1,2,3,4....starting with 0
        """
        A = np.log(self.A * 10) # for scaling purpose multiplication, log of small number will be a very large number
        B = np.log(self.B * 100)
        pie = np.log(self.pie * 10)
        best_state = np.zeros(shape=[A.shape[0], obs.shape[0]], dtype=np.int32)
        best_state[:, :] = -1
        best_prob1 = pie + B[:, obs[0]]
        # print("initially best_prob1 is", best_prob1)
        for time in range(1, obs.shape[0]):
            best_prob2 = np.zeros(best_prob1.shape)
            #        print("best_prob1 is",best_prob1)
            for i in range(A.shape[0]):
                p = best_prob1 + A[:, i]
                # print(p.shape)
                best_prob2[i] = np.max(p)
                best_state[i, time] = np.argmax(p)
            best_prob2 = best_prob2 + B[:, obs[time]]
            best_prob1 = best_prob2
            # print("best_prob1 is", best_prob1)
        state_sequence = [np.argmax(best_prob1)]
        i = obs.shape[0] - 1
        while i > 0:
            state_sequence.append(best_state[state_sequence[-1]][i])
            i -= 1
            # print(state_sequence)
        state_sequence.reverse()
        # print('state sequence is',state_sequence)
        return best_state, state_sequence

    # def forward_scaled(self,obs):
    #     """
    #     :param obs: is a 1d array shape (T,), observations are in range(0,num_obs)
    #     :return: calculate alphas of the forward calculation, shape(alpha) = T X num_states
    #     """
    #     self.alpha_tilda = np.zeros(shape=(obs.shape[0], self.num_states))
    #     self.alpha_tilda[0] = self.pie * self.B[:, obs[0]]
    #     self.scale_factor = [np.sum(self.alpha_tilda[0])]
    #     # alpha_tilda[0] = sc
    #     # print(scale_factor)
    #     for i in range(1, obs.shape[0]):
    #         #        print(alpha[i-1])
    #         #        print(A)
    #         self.alpha_tilda[i - 1] /= self.scale_factor[i - 1]
    #         temp = np.matmul(self.alpha_tilda[i - 1], self.A)
    #         #        print("temp is",temp)
    #         self.alpha_tilda[i] = temp * self.B[:, obs[i]]
    #         self.scale_factor.append(np.sum(self.alpha_tilda[i]))
    #     self.alpha_tilda[-1] /= self.scale_factor[-1]
    #     # return self.alpha_tilda, self.scale_factor

    # def backward_scaled(self, obs):
    #     self.beta_tilda = np.zeros(shape=(obs.shape[0], self.A.shape[0]))
    #     self.beta_tilda[obs.shape[0] - 1] = 1
    #     # self.beta_tilda /= self.scale_factor[-1]
    #     time = obs.shape[0] - 2
    #     while time >= 0:
    #         self.beta_tilda[time + 1] /= self.scale_factor[time + 1]
    #         for i in range(self.A.shape[0]):
    #             self.beta_tilda[time][i] = np.sum(self.beta_tilda[time + 1] * self.A[i] * self.B[:, obs[time + 1]])
    #             # beta[time] /= scale_factor[time]
    #         time -= 1
    #     self.beta_tilda[0] /= self.scale_factor[0]

    # def train_on_examples_scaled(self, states, observations):
    #     """
    #     :param states: list of 1d array each of shape = (T, )
    #     :param observations:list of 1d array each of shape = (T, )
    #     :return:
    #     """
    #     pie_temp = np.zeros(self.pie.shape)
    #     A_num_temp = np.zeros(self.A.shape)
    #     A_deno_temp = np.zeros((self.num_states,1))
    #     B_num_temp = np.zeros(self.B.shape)
    #     B_deno_temp = np.zeros((self.A.shape[0],1))
    #
    #     log_prob = 0
    #     for obs in observations:
    #         self.forward_scaled(obs)
    #         # log_prob += np.log(np.sum(self.alpha[-1])) #modify this
    #         log_prob += np.log(np.prod(self.scale_factor))
    #         self.backward_scaled(obs)
    #         zeta, gamma = [], []
    #         for t in range(obs.shape[0]-1):
    #             ## calculate zeta_t
    #             # zeta_table_for_t = np.zeros(self.A.shape[0], self.A.shape[0])
    #             temp = self.beta_tilda[t + 1] * self.B[:, obs[t + 1]]
    #             # temp = np.reshape(temp, newshape=(1, temp.shape[0]))
    #             temp2 = self.alpha_tilda[t]
    #             temp3 = np.outer(temp2, temp)
    #             zeta_table_for_t = temp3 * self.A
    #             zeta_table_for_t /= np.sum(zeta_table_for_t)
    #             zeta.append(zeta_table_for_t)
    #             ## calculate gamma
    #             gamma_t = np.sum(zeta_table_for_t, axis=1)
    #             gamma.append(gamma_t)
    #
    #         prob_of_obs = np.sum(self.alpha_tilda[-1]) #modify this
    #         # prob_of_obs = np.prod(self.scale_factor)
    #         ###updates...
    #         zeta = np.array(zeta)
    #         gamma = np.array(gamma)
    #         pie_temp += gamma[0]
    #         temp = np.sum(gamma, axis = 0).reshape(self.num_states,1)
    #         A_num_temp += np.sum(zeta, axis =0)/prob_of_obs
    #         A_deno_temp += temp/(prob_of_obs + 1e-6)
    #
    #         # updating B
    #         # modify gamma array, now calculate gamma_T
    #         gamma_T = self.alpha_tilda[-1,:]/np.sum(self.alpha_tilda[-1,:]) # gamma of last time step
    #         gamma = gamma.tolist()
    #         gamma.append(gamma_T)
    #         gamma = np.array(gamma)
    #         B_deno_temp += np.sum(gamma, axis = 0).reshape(-1,1) # required for broadcasting along column
    #         # B_ = np.zeros(self.B.shape)
    #         for i in range(obs.shape[0]):
    #             B_num_temp[:,obs[i]] += gamma[i]
    #             # print('B_ is',B_)
    #         B_num_temp /= (prob_of_obs + 1e-6)
    #         B_deno_temp /= (prob_of_obs + 1e-6)
    #
    #     self.A= A_num_temp / A_deno_temp
    #     self.B = B_num_temp / B_deno_temp
    #     self.pie = pie_temp/np.sum(pie_temp)
    #     # print("A is", self.A)
    #     print('A row sum is',np.sum(self.A, axis=1))
    #     # print('B is', self.B)
    #     print('B row sum is',np.sum(self.B, axis = 1))
    #     print('pie is',self.pie)
    #     print('initial log prob=',log_prob)
    #     return log_prob

    def forward(self,obs):
        """
        :param obs: is a 1d array shape (T,), observations are in range(0,num_obs)
        :return: calculate alphas of the forward calculation, shape(alpha) = T X num_states
        """
        self.alpha = np.zeros(shape=(obs.shape[0], self.num_states))
        self.alpha[0] = self.pie * self.B[:, obs[0]]
        # scale_factor = [np.sum(alpha[0])]
        # print(scale_factor)
        for i in range(1, obs.shape[0]):
            #        print(alpha[i-1])
            #        print(A)
            # alpha[i - 1] /= scale_factor[i - 1]
            temp = np.matmul(self.alpha[i - 1], self.A)
            #        print("temp is",temp)
            self.alpha[i] = temp * self.B[:, obs[i]]
            # scale_factor.append(np.sum(alpha[i]))

    def backward(self, obs):
        self.beta = np.zeros(shape=(obs.shape[0], self.A.shape[0]))
        self.beta[obs.shape[0] - 1] = 1
        time = obs.shape[0] - 2
        while time >= 0:
            for i in range(self.A.shape[0]):
                #            beta[time+1] /= scale_factor[time+1]
                self.beta[time][i] = np.sum(self.beta[time + 1] * self.A[i] * self.B[:, obs[time + 1]])
                # beta[time] /= scale_factor[time]
            time -= 1
    def generate_sequence(self, maxlen = 5):
        '''
        :param maxlen: maximum length of sequence to bw generated
        :return:
        '''
        np.random.seed(1)
        gen_sequence = []
        gen_states = []
        gen_states.append(np.argmax(self.pie)) # initial state

        for i in range(maxlen):
            gen_states.append(np.argmax(np.random.multinomial(1,self.A[gen_states[-1]]))) # sampling
            gen_sequence.append(np.argmax(np.random.multinomial(1,self.B[gen_states[-1]]))) #sampling

            if gen_sequence[-1] == self.num_obs-1: # checking if end symbol was genrarated
                pass
                break

        return gen_sequence,gen_states

    # def train_on_one_example(self, obs, state=None):
    #     """
    #     :param state: 1 d array , shape = (T, )
    #     :param obs: 1d array, shape = (T, )
    #     :return:
    #     """
    #     self.forward(obs)
    #     self.backward(obs)
    #     zeta, gamma = [], []
    #     for t in range(self.num_states-1):
    #         ## calculate zeta_t
    #         # zeta_table_for_t = np.zeros(self.A.shape[0], self.A.shape[0])
    #         temp = self.beta[t + 1] * self.B[:, obs[t + 1]]
    #         # temp = np.reshape(temp, newshape=(1, temp.shape[0]))
    #         temp2 = self.alpha[t]
    #         temp3 = np.outer(temp2, temp)
    #         zeta_table_for_t = temp3 * self.A
    #         zeta_table_for_t /= np.sum(zeta_table_for_t)
    #         zeta.append(zeta_table_for_t)
    #
    #         ## calculate gamma
    #         gamma_t = np.sum(zeta_table_for_t, axis=1)
    #         gamma.append(gamma_t)
    #
    #     ###updates...
    #     zeta = np.array(zeta)
    #     gamma = np.array(gamma)
    #     self.pie = gamma[0]
    #     temp = np.sum(gamma, axis = 0).reshape(self.num_states,1)
    #     self.A = np.sum(zeta, axis =0) / temp
    #
    #     print("A is", self.A)
    #     # updating B
    #     # modify gamma array, now calculate gamma_T
    #     gamma_T = self.alpha[-1,:]/np.sum(self.alpha[-1,:]) # gamma of last time step
    #     gamma = gamma.tolist()
    #     gamma.append(gamma_T)
    #     gamma = np.array(gamma)
    #     # print('gamma is', gamma)
    #
    #     # indicator_3d = np.zeros((self.num_obs, gamma.shape[0], gamma.shape[1]))
    #     # # assuming obs states are 0,1,2...., obs-1
    #     # for i in range(self.num_obs):
    #     #     for j in range(obs.shape[0]):
    #     #         if obs[j] == i:
    #     #             indicator_3d[i,j,state[j]] += 1
    #     # print(indicator_3d)
    #     # exit()
    #     # temp = temp.reshape((temp.shape[0],))
    #     deno_B = np.sum(gamma, axis = 0).reshape(-1,1) # required for broadcasting along column
    #     B_ = np.zeros(self.B.shape)
    #     for i in range(obs.shape[0]):
    #         B_[:,obs[i]] += gamma[i]
    #         # print('B_ is',B_)
    #     B_/= deno_B
    #     self.B = B_
    #     # print("B is", self.B)
    #     # print('sum of emission prob', np.sum(self.B, axis=1))

    def train_on_examples(self, observations, states=None, verbose = True):
        """
        :param states: list of 1d array each of shape = (T, )
        :param observations:list of 1d array each of shape = (T, )
        :return:
        """
        pie_temp = np.zeros(self.pie.shape)
        A_num_temp = np.zeros(self.A.shape)
        A_deno_temp = np.zeros((self.num_states,1))
        B_num_temp = np.zeros(self.B.shape)
        B_deno_temp = np.zeros((self.A.shape[0],1))

        log_prob = 0
        for obs in observations:
            self.forward(obs)
            log_prob += np.log(np.sum(self.alpha[-1]))
            self.backward(obs)
            zeta_num = np.zeros((obs.shape[0] - 1, self.num_states, self.num_states))
            gamma_num = self.alpha * self.beta
            # gamma /= np.sum(self.alpha[-1]) # is not needed, since it is later divided by probability

            for t in range(obs.shape[0]-1):
                ## calculate zeta_t
                # zeta_table_for_t = np.zeros(self.A.shape[0], self.A.shape[0])
                # temp = self.beta[t + 1] * self.B[:, obs[t + 1]]
                # temp = np.reshape(temp, newshape=(1, temp.shape[0]))
                # temp2 = self.alpha[t]
                # temp3 = np.outer(self.alpha[t], self.beta[t + 1] * self.B[:, obs[t + 1]])
                zeta_num[t] = np.outer(self.alpha[t], self.beta[t + 1] * self.B[:, obs[t + 1]]) * self.A
                # zeta[t] /= np.sum(zeta[t])# is not needed, since it is being divided by probability later
                # zeta.append(zeta_table_for_t)
                ## calculate gamma
                # gamma_t = np.sum(zeta_table_for_t, axis=1)
                # gamma.append(gamma_t)

            prob_of_obs = np.sum(self.alpha[-1])
            ###updates...
            # zeta = np.array(zeta)
            # gamma = np.array(gamma)
            pie_temp += (gamma_num[0]/prob_of_obs)
            temp = np.sum(gamma_num[0:-1], axis = 0).reshape(self.num_states,1)
            A_num_temp += np.sum(zeta_num, axis =0)/(prob_of_obs )
            A_deno_temp += temp/(prob_of_obs ) # can add a small number 1e-6

            # updating B
            # modify gamma array, now calculate gamma_T
            # gamma_T = self.alpha[-1,:]/np.sum(self.alpha[-1,:]) # gamma of last time step, since beta is all 1s
            # gamma = gamma.tolist()
            # gamma.append(gamma_T)
            # gamma = np.array(gamma)
            B_deno_temp += (np.sum(gamma_num, axis = 0).reshape(-1,1)/prob_of_obs) # required for broadcasting along column
            # B_ = np.zeros(self.B.shape)
            for i in range(obs.shape[0]):
                B_num_temp[:,obs[i]] += (gamma_num[i]/prob_of_obs)

            print('prob of obs is', prob_of_obs)
        self.A= A_num_temp / A_deno_temp
        self.B = B_num_temp / B_deno_temp
        self.pie = pie_temp / np.sum(pie_temp)
        # self.pie = pie_temp/np.sum(pie_temp)
        if verbose:
            print("A is", self.A)
            print('A sum along row is',np.sum(self.A, axis=1))
            print('B is', self.B)
            print('B sum along row',np.sum(self.B, axis = 1))
            print('pie is',self.pie)
            print('initial log prob=',log_prob)
        return log_prob

class hmm_scale(hmm):
    def __init__(self,num_states, num_obs, states=None, observations=None, A=None, B= None, pie=None):
        super().__init__(num_states, num_obs, states, observations, A, B, pie)
    def forward(self,obs):
        """
                :param obs: is a 1d array shape (T,), observations are in range(0,num_obs)
                :return: calculate alphas of the forward calculation, shape(alpha) = T X num_states
                """
        self.alpha_tilda = np.zeros(shape=(obs.shape[0], self.num_states))
        self.alpha_tilda[0] = self.pie * self.B[:, obs[0]]
        self.scale_factor = [np.sum(self.alpha_tilda[0])]
        # alpha_tilda[0] = sc
        # print(scale_factor)
        for i in range(1, obs.shape[0]):
            #        print(alpha[i-1])
            #        print(A)
            self.alpha_tilda[i - 1] /= self.scale_factor[i - 1]
            temp = np.matmul(self.alpha_tilda[i - 1], self.A)
            #        print("temp is",temp)
            self.alpha_tilda[i] = temp * self.B[:, obs[i]]
            self.scale_factor.append(np.sum(self.alpha_tilda[i]))
        self.alpha_tilda[-1] /= self.scale_factor[-1]
        # return self.alpha_tilda, self.scale_factor
    def backward(self, obs):
        self.beta_tilda = np.zeros(shape=(obs.shape[0], self.A.shape[0]))
        self.beta_tilda[obs.shape[0] - 1] = 1
        # self.beta_tilda /= self.scale_factor[-1]
        time = obs.shape[0] - 2
        while time >= 0:
            self.beta_tilda[time + 1] /= self.scale_factor[time + 1]
            for i in range(self.A.shape[0]):
                self.beta_tilda[time][i] = np.sum(self.beta_tilda[time + 1] * self.A[i] * self.B[:, obs[time + 1]])
                # beta[time] /= scale_factor[time]
            time -= 1
        self.beta_tilda[0] /= self.scale_factor[0]

    def train_on_examples(self, observations, states=None, verbose = True):
        """
        :param states: list of 1d array each of shape = (T, )
        :param observations:list of 1d array each of shape = (T, )
        :return:
        """
        pie_temp = np.zeros(self.pie.shape)
        A_num_temp = np.zeros(self.A.shape)
        A_deno_temp = np.zeros((self.num_states,1))
        B_num_temp = np.zeros(self.B.shape)
        B_deno_temp = np.zeros((self.A.shape[0],1))

        log_prob = 0
        sum_prob = 0
        for obs in observations:
            self.forward(obs)
            # log_prob += np.log(np.sum(self.alpha[-1])) #modify this
            log_prob += np.log(np.prod(self.scale_factor))
            sum_prob += np.prod(self.scale_factor) # checking this since sequence length is very large > 30..going to zero prob
            # print('prob is',sum_prob)
            self.backward(obs)
            # print('alpha_tilda', self.alpha_tilda)
            # print('beta tilda', self.beta_tilda)
            gamma_num = self.alpha_tilda * self.beta_tilda

            ''' no where explained/mentioned in book
            is necessary in scaled version
            '''
            gamma_num *= np.asarray(self.scale_factor).reshape(-1,1) # is necessary, debugging this killed me.
            # print('gamma num',gamma_num)
            zeta_num = np.zeros((obs.shape[0] - 1, self.num_states, self.num_states))

            for t in range(obs.shape[0]-1):
                zeta_num[t] = np.outer(self.alpha_tilda[t], self.beta_tilda[t + 1] * self.B[:, obs[t + 1]]) * self.A

            # print('zeta num sum along 2nd axis',np.sum(zeta_num,axis=2))
            # prob_of_obs = np.sum(self.alpha_tilda[-1]) #modify this
            prob_of_obs = np.prod(self.scale_factor)
            print('prob_of_obs is', prob_of_obs)
            prob_of_obs = 1
            ###updates...
            pie_temp += gamma_num[0]
            temp = np.sum(gamma_num[0:-1], axis=0).reshape(self.num_states, 1)
            '''can we do this division on log scale..???
            '''
            A_num_temp += np.sum(zeta_num, axis=0)
            A_deno_temp += temp
            # print('zeta num', zeta_num)
            # updating B
            # modify gamma array, now calculate gamma_T
            # gamma_T = self.alpha_tilda[-1,:]/np.sum(self.alpha_tilda[-1,:]) # gamma of last time step
            # gamma = gamma.tolist()
            # gamma.append(gamma_T)
            # gamma = np.array(gamma)
            B_deno_temp += (np.sum(gamma_num, axis=0).reshape(-1, 1) )  # required for broadcasting along column
            # B_ = np.zeros(self.B.shape)
            for i in range(obs.shape[0]):
                B_num_temp[:, obs[i]] += gamma_num[i]

            # pie_temp += (gamma_num[0]/prob_of_obs)
            # temp = np.sum(gamma_num[0:-1], axis = 0).reshape(self.num_states,1)
            # '''can we do this division on log scale..???
            # '''
            # A_num_temp += np.sum(zeta_num, axis =0)/(prob_of_obs ) # can add 1 e-6 in denominator
            # A_deno_temp += temp/(prob_of_obs) # can add 1 e-6 in denominator
            # # print('zeta num', zeta_num)
            # # updating B
            # # modify gamma array, now calculate gamma_T
            # # gamma_T = self.alpha_tilda[-1,:]/np.sum(self.alpha_tilda[-1,:]) # gamma of last time step
            # # gamma = gamma.tolist()
            # # gamma.append(gamma_T)
            # # gamma = np.array(gamma)
            # B_deno_temp += (np.sum(gamma_num, axis = 0).reshape(-1,1)/(prob_of_obs)) # required for broadcasting along column
            # # B_ = np.zeros(self.B.shape)
            # for i in range(obs.shape[0]):
            #     B_num_temp[:,obs[i]] += (gamma_num[i]/prob_of_obs)
                # print('B_ is',B_)
            # B_num_temp /= (prob_of_obs ) # can add 1 e-6 in denominator
            # B_deno_temp /= (prob_of_obs )# can add 1 e-6 in denominator
            # print('prob of obs is', np.prod(self.scale_factor))

        # print('A denominator', A_deno_temp)
        # print('B denominator', B_deno_temp)
        self.A= A_num_temp / A_deno_temp
        self.B = B_num_temp / B_deno_temp
        self.pie = pie_temp / np.sum(pie_temp)
        if verbose:
            print("A is", self.A)
            print('sum along row for A is',np.sum(self.A, axis=1))
            print('B is', self.B)
            print('sum of row for B is',np.sum(self.B, axis = 1))
            print('pie is',self.pie)
            print('sum of pie',np.sum(self.pie))
            print('initial log prob=',log_prob)

        return log_prob

class hmm_scale_start_state(hmm_scale):
    '''
    specific start and end state
    fixed starting state, implies
    no need to update pie, smoothening not required on first row of A matrix
    '''
    def __init__(self,num_states, num_obs, states=None, observations=None, A=None, B= None, pie=None):
        super().__init__(num_states, num_obs, states, observations, A, B, pie)

    def initialise(self, states, observations):
        """
        :param states: list of state sequences each a np array
        :param obsverations: list of observation sequences each a np array
        counting the occurrence and normalizing to probability
        """
        # implement initialisation here
        self.A = np.ones((self.num_states, self.num_states)) # laplacian smoothening
        self.pie = np.zeros((self.num_states,) )
        self.pie[0] = 1.0

        for s in states:
            for i in range(s.shape[0]-1):
                self.A[s[i]][s[i+1]] += 1
        denominator = np.sum(self.A, axis =1)
        denominator = denominator.reshape(-1,1)
        self.A /= denominator

        self.B = np.ones((self.num_states, self.num_obs)) # laplacian kind of smoothening
        for j in range(len(observations)):
            for i in range(observations[j].shape[0]):
                self.B[states[j][i]][observations[j][i]] += 1
        denominator = np.sum(self.B, axis =1)
        denominator = denominator.reshape(-1,1)
        self.B /= denominator

    def viterbiAlgorithm(self, obs):
        """
        modify it to adjust for fixed initial state
        obs is a 1d array assumed it contains 0,1,2,3,4....starting with 0
        """
        A = self.A # for scaling purpose multiplication, log of small number will be a very large number
        B = self.B
        pie = self.pie
        best_state = np.zeros(shape=[A.shape[0], obs.shape[0]], dtype=np.int32)
        best_state[:, :] = -1
        best_prob1 = pie * B[:, obs[0]]
        # print("initially best_prob1 is", best_prob1)
        for time in range(1, obs.shape[0]):
            best_prob2 = np.zeros(best_prob1.shape)
            #        print("best_prob1 is",best_prob1)
            for i in range(A.shape[0]):
                p = best_prob1 * A[:, i]
                # print(p.shape)
                best_prob2[i] = np.max(p)
                best_state[i, time] = np.argmax(p)
            best_prob2 = best_prob2 * B[:, obs[time]]
            best_prob1 = best_prob2
            # print("best_prob1 is", best_prob1)
        state_sequence = [np.argmax(best_prob1)]
        i = obs.shape[0] - 1
        while i > 0:
            state_sequence.append(best_state[state_sequence[-1]][i])
            i -= 1
            # print(state_sequence)
        state_sequence.reverse()
        # print('state sequence is',state_sequence)
        return best_state, state_sequence, np.max(best_prob1)

    def train_on_examples_scaled(self, observations, states= None):
        """
        :param states: list of 1d array each of shape = (T, )
        :param observations:list of 1d array each of shape = (T, )
        :return:
        """
        # pie_temp = np.zeros(self.pie.shape)
        A_num_temp = np.zeros(self.A.shape)
        A_deno_temp = np.zeros((self.num_states, 1))
        B_num_temp = np.zeros(self.B.shape)
        B_deno_temp = np.zeros((self.A.shape[0], 1))

        log_prob = 0
        for obs in observations:
            self.forward(obs)
            # log_prob += np.log(np.sum(self.alpha[-1])) #modify this
            log_prob += np.log(np.prod(self.scale_factor))
            self.backward(obs)
            zeta, gamma = [], []
            for t in range(obs.shape[0] - 1):
                ## calculate zeta_t
                # zeta_table_for_t = np.zeros(self.A.shape[0], self.A.shape[0])
                temp = self.beta_tilda[t + 1] * self.B[:, obs[t + 1]]
                # temp = np.reshape(temp, newshape=(1, temp.shape[0]))
                temp2 = self.alpha_tilda[t]
                temp3 = np.outer(temp2, temp)
                zeta_table_for_t = temp3 * self.A
                zeta_table_for_t /= np.sum(zeta_table_for_t)
                zeta.append(zeta_table_for_t)
                ## calculate gamma
                gamma_t = np.sum(zeta_table_for_t, axis=1)
                gamma.append(gamma_t)

            prob_of_obs = np.sum(self.alpha_tilda[-1])  # modify this
            # prob_of_obs = np.prod(self.scale_factor)
            ###updates...
            zeta = np.array(zeta)
            gamma = np.array(gamma)
            # pie_temp += gamma[0]
            temp = np.sum(gamma, axis=0).reshape(self.num_states, 1)
            A_num_temp += np.sum(zeta, axis=0) / prob_of_obs
            A_deno_temp += temp / (prob_of_obs + 1e-6)

            # updating B
            # modify gamma array, now calculate gamma_T
            gamma_T = self.alpha_tilda[-1, :] / np.sum(self.alpha_tilda[-1, :])  # gamma of last time step
            gamma = gamma.tolist()
            gamma.append(gamma_T)
            gamma = np.array(gamma)
            B_deno_temp += np.sum(gamma, axis=0).reshape(-1, 1)  # required for broadcasting along column
            # B_ = np.zeros(self.B.shape)
            for i in range(obs.shape[0]):
                B_num_temp[:, obs[i]] += gamma[i]
                # print('B_ is',B_)
            B_num_temp /= (prob_of_obs + 1e-6)
            B_deno_temp /= (prob_of_obs + 1e-6)

        self.A = A_num_temp / A_deno_temp
        self.B = B_num_temp / B_deno_temp
        # self.pie = pie_temp / np.sum(pie_temp)
        # print("A is", self.A)
        print('A row sum is',np.sum(self.A, axis=1))
        # print('B is', self.B)
        print('B row sum is',np.sum(self.B, axis=1))
        print('pie is', self.pie)
        print('initial log prob=', log_prob)
        return log_prob

    # def generate_sequence(self, maxlen = 5):
    #     '''
    #     :param maxlen: maximum length of sequence to bw generated
    #     :return:
    #     '''
    #     np.random.seed(1)
    #     gen_sequence = []
    #     gen_states = []
    #     gen_states.append(0) # initial state
    #
    #     for i in range(maxlen):
    #         gen_states.append(np.argmax(np.random.multinomial(1,self.A[gen_states[-1]])))
    #         gen_sequence.append(np.argmax(np.random.multinomial(1,self.B[gen_states[-1]])))
    #
    #         if gen_sequence[-1] == self.num_obs-1: # checking if end symbol was genrarated
    #             pass
    #             break
    #
    #     return gen_sequence

    def generate_sequence_using_sampling(self, maxlen=5, n_opt_step = 2):
        '''
        :param maxlen:
        :param n_opt_step: number of options to consider at each step, can be calculated as well
        :return: from all possible options, chose the one that has maximum probability
        '''
        # how to store all such options, try using dictionary
        np.random.seed(1)
        pass
