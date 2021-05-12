import numpy as np
from State import State
from Action import Action

'''
Includes blood glucose level proxy for diabetes: 0-3
    (lo2, lo1, normal, hi1, hi2); Any other than normal is "abnormal"
Initial distribution:
    [.05, .15, .6, .15, .05] for non-diabetics and [.01, .05, .15, .6, .19] for diabetics

Effect of vasopressors on if diabetic:
    raise blood pressure: normal -> hi w.p. .9, lo -> normal w.p. .5, lo -> hi w.p. .4
    raise blood glucose by 1 w.p. .5

Effect of vasopressors off if diabetic:
    blood pressure falls by 1 w.p. .05 instead of .1
    glucose does not fall - apply fluctuations below instead

Fluctuation in blood glucose levels (IV/insulin therapy are not possible actions):
    fluctuate w.p. .3 if diabetic
    fluctuate w.p. .1 if non-diabetic
Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4530321/

Additional fluctuation regardless of other changes
This order is applied:
    antibiotics, ventilation, vasopressors, fluctuations
'''

class MDP(object):

    def __init__(self, init_state_idx=None, init_state_idx_type='obs',
            policy_array=None, policy_idx_type='obs', p_diabetes=0.2):
        '''
        initialize the simulator
        '''
        assert p_diabetes >= 0 and p_diabetes <= 1, \
                "Invalid p_diabetes: {}".format(p_diabetes)
        assert policy_idx_type in ['obs', 'full', 'proj_obs']

        # Check the policy dimensions (states x actions)
        if policy_array is not None:
            assert policy_array.shape[1] == Action.NUM_ACTIONS_TOTAL
            if policy_idx_type == 'obs':
                assert policy_array.shape[0] == State.NUM_OBS_STATES
            elif policy_idx_type == 'full':
                assert policy_array.shape[0] == \
                        State.NUM_HID_STATES * State.NUM_OBS_STATES
            elif policy_idx_type == 'proj_obs':
                assert policy_array.shape[0] == State.NUM_PROJ_OBS_STATES

        # p_diabetes is used to generate random state if init_state is None
        self.p_diabetes = p_diabetes
        self.state = None
        # Only need to use init_state_idx_type if you are providing a state_idx!
        self.state = self.get_new_state(init_state_idx, init_state_idx_type)

        self.policy_array = policy_array
        self.policy_idx_type = policy_idx_type  # Used for mapping the policy to actions

    def get_new_state(self, state_idx = None, idx_type = 'obs', diabetic_idx = None):
        '''
        use to start MDP over.  A few options:

        Full specification:
        1. Provide state_idx with idx_type = 'obs' + diabetic_idx
        2. Provide state_idx with idx_type = 'full', diabetic_idx is ignored
        3. Provide state_idx with idx_type = 'proj_obs' + diabetic_idx*

        * This option will set glucose to a normal level

        Random specification
        4. State_idx, no diabetic_idx: Latter will be generated
        5. No state_idx, no diabetic_idx:  Completely random
        6. No state_idx, diabetic_idx given:  Random conditional on diabetes
        '''
        assert idx_type in ['obs', 'full', 'proj_obs']
        option = None
        if state_idx is not None:
            if idx_type == 'obs' and diabetic_idx is not None:
                option = 'spec_obs'
            elif idx_type == 'obs' and diabetic_idx is None:
                option = 'spec_obs_no_diab'
                diabetic_idx = np.random.binomial(1, self.p_diabetes)
            elif idx_type == 'full':
                option = 'spec_full'
            elif idx_type == 'proj_obs' and diabetic_idx is not None:
                option = 'spec_proj_obs'
        elif state_idx is None and diabetic_idx is None:
            option = 'random'
        elif state_idx is None and diabetic_idx is not None:
            option = 'random_cond_diab'

        assert option is not None, "Invalid specification of new state"

        if option in ['random', 'random_cond_diab'] :
            init_state = self.generate_random_state(diabetic_idx)
            # Do not start in death or discharge state
            while init_state.check_absorbing_state():
                init_state = self.generate_random_state(diabetic_idx)
        else:
            # Note that diabetic_idx will be ignored if idx_type = 'full'
            init_state = State(
                    state_idx=state_idx, idx_type=idx_type,
                    diabetic_idx=diabetic_idx)

        return init_state

    def generate_random_state(self, diabetic_idx=None):
        # Note that we will condition on diabetic idx if provided
        if diabetic_idx is None:
            diabetic_idx = np.random.binomial(1, self.p_diabetes)

        # hr and sys_bp w.p. [.25, .5, .25]
        hr_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        sysbp_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        # percoxyg w.p. [.2, .8]
        percoxyg_state = np.random.choice(np.arange(2), p=np.array([.2, .8]))

        if diabetic_idx == 0:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.05, .15, .6, .15, .05]))
        else:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.01, .05, .15, .6, .19]))
        antibiotic_state = 0
        vaso_state = 0
        vent_state = 0

        state_categs = [hr_state, sysbp_state, percoxyg_state,
                glucose_state, antibiotic_state, vaso_state, vent_state]
        # DEFAULT TO SOME STATE (ie. 2 abnormal states)
        # state_categs = [0, 0, 1, 2, antibiotic_state, vaso_state, vent_state]
        return State(state_categs=state_categs, diabetic_idx=diabetic_idx)

    def transition_antibiotics_on(self, probs):
        '''
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .5
        '''
        self.state.antibiotic_state = 1
        
        temp = np.zeros((720,))
        temp_probs = []

        for (idx, pr) in enumerate(probs):
            if pr > 0:
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)

                r = 0.5
                if self.state.hr_state == 2:
                    temp_probs.append((idx, pr *r))
                    self.state.hr_state = 1
                    temp_probs.append((self.state.get_state_idx(), pr*r))
                else:
                    temp_probs.append((idx, pr))

        for (x, y) in temp_probs:
            if self.state.sysbp_state == 2:
                self.state.set_state_by_idx(x, idx_type='obs', diabetic_idx=0)
                temp[self.state.get_state_idx()] += y * r
                self.state.sysbp_state = 1
                temp[self.sate.get_state_idx()] += y *r
            else:
                temp[x] = y

        # assert(sum(temp) == 1.)
        return temp


    def transition_antibiotics_off(self, probs):
        '''
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        '''
        def set_new_idx(temp, pr):
            if self.state.antibiotic_state == 1:
                r = 0.1
                probs = []
                if self.state.hr_state == 1: # w.p. r
                    probs.append((self.state.get_state_idx(), pr * (1-r)))
                    self.state.hr_state = 2
                    probs.append((self.state.get_state_idx(), pr * r))
                else:
                    probs.append((self.state.get_state_idx(), pr))
                new_probs = []
                if self.state.sysbp_state == 1: # w.p. r
                    for (x, y) in probs:
                        self.state.set_state_by_idx(x, idx_type='obs', diabetic_idx=0)
                        new_probs.append((self.state.get_state_idx(), y * (1-r)))
                        self.state.sysbp_state = 2
                        new_probs.append((self.state.get_state_idx(), y * r))
                else:
                    new_probs = probs.copy()

                for (x, y) in new_probs:
                    self.state.set_state_by_idx(x, idx_type='obs', diabetic_idx=0)
                    self.state.antibiotic_state = 0
                    new_idx = self.state.get_state_idx()
                    temp[new_idx] += y 
            
            else:
                self.state.antibiotic_state = 0
                idx = self.state.get_state_idx()
                temp[idx] += pr
            return temp

        temp = np.zeros((720,))
        for (idx, pr) in enumerate(probs):
            if pr > 0:
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                temp = set_new_idx(temp, pr)
        # assert(sum(temp) == 1.0)
        return temp

    def transition_vent_on(self, probs):
        '''
        ventilation state on
        percent oxygen: low -> normal w.p. .7
        '''
        def set_new_state(temp, pr):
            r = 0.7
            cur_idx = self.state.get_state_idx()
            if self.state.percoxyg_state == 0:
                self.state.percoxyg_state = 1
                temp_idx = self.state.get_state_idx()
                temp[temp_idx] += pr * r
            temp[cur_idx] += pr * (1-r)
            return temp

        self.state.vent_state = 1
        temp = np.zeros((720,))
        for (idx, pr) in enumerate(probs):
            if pr > 0:
                self.state_set_by_idx(idx)
                temp = set_new_state(temp, pr)
        return temp

    def transition_vent_off(self, probs):
        '''
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        '''
        def set_new_idx(temp, next_pr):
            temp_id = self.state.get_state_idx()
            temp[temp_id] += next_pr
            return temp

        def get_next_state_probs(temp, pr):
            if self.state.vent_state == 1 and self.state.percoxyg_state == 1:
                r = 0.1
                self.state.vent_state = 0
                temp = set_new_idx(temp, pr * (1-r))
                self.state.percoxyg_state = 0
                temp = set_new_idx(temp, pr * r)
            else:        
                self.state.vent_state = 0
                temp = set_new_idx(temp, pr)
            return temp
        
        temp = np.zeros((720,))
        for (idx, pr) in enumerate(probs):
            if pr > 0:
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                temp = get_next_state_probs(temp, pr)
        
        # assert(sum(temp) == 1.0)
        return temp

    def transition_vaso_on(self, probs):
        '''
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .5
        '''
        def set_new_idx(temp, next_pr):
            temp_id = self.state.get_state_idx()
            temp[temp_id] += next_pr
            return temp

        def get_next_state_probs(temp, pr):
            self.state.vaso_state = 1
            if self.state.diabetic_idx == 0:
                dib_r = 0.7
                if self.state.sysbp_state == 0:
                    temp = set_new_idx(temp, pr * (1-dib_r))
                    self.state.sysbp_state = 1
                    temp = set_new_idx(temp, pr * dib_r)

                elif self.state.sysbp_state == 1:
                    temp = set_new_idx(temp, pr * (1-dib_r))
                    self.state.sysbp_state = 2
                    temp = set_new_idx(temp, pr * (1-dib_r))
                return temp

        temp = np.zeros((720,))
        for (idx, pr) in enumerate(probs):
            if pr > 0 :
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                temp = get_next_state_probs(temp, pr)
            # else:
                # print("non zero diabetic idx")
                # assert(False)
        # assert(sum(temp) == 1.)
        return temp
        '''
            if self.state.sysbp_state == 1:
                sys_r = 0.9
                if np.random.uniform(0,1) < sys_r:
                    self.state.sysbp_state = 2
            elif self.state.sysbp_state == 0:
                up_prob = np.random.uniform(0,1)
                r1= 0.5
                r2 = 0.9
                if up_prob < r1:
                    self.state.sysbp_state = 1
                elif up_prob < r2:
                    self.state.sysbp_state = 2
            g_r = 0.5
            if np.random.uniform(0,1) < g_r:
                self.state.glucose_state = min(4, self.state.glucose_state + 1)
        '''

    def transition_vaso_off(self, probs):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        def set_new_idx(temp, pr):
            if self.state.diabetic_idx == 0:
                r = 0.1
                temp_idx = self.state.get_state_idx()
                self.state.vaso_state = 0
                temp[self.state.get_state_idx()] += pr * (1-r)
                
                self.state.set_state_by_idx(temp_idx, idx_type='obs', diabetic_idx=0)
                self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
                self.state.vaso_state = 0
                temp[self.state.get_state_idx()] += pr * r
                
            else:
                print("Diabetix idx should always be 0!")
                assert(False)
                '''
                r = 0.05
                if np.random.uniform(0,1) < r:
                    self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
                '''
            return temp

        temp = np.zeros((720,))

        for (idx, pr) in enumerate(probs):
            if pr > 0:
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                if self.state.vaso_state == 1:
                    temp = set_new_idx(temp, pr)
                else:
                    temp[idx] += pr
        # print(temp)
        # assert(sum(temp) == 1)
        return temp

    def _next_transition(self, old, state, rate):
        temp = []
        for (idx, pr) in old:
            self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
            if state == 'hr':
                self.state.hr_state = max(0, self.state.hr_state - 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                self.state.hr_state = min(2, self.state.hr_state + 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                temp.append((idx, pr * (1 - 2 * rate)))
            elif state == 'sysbp':
                self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                self.state.sysbp_state = min(2, self.state.sysbp_state + 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                temp.append((idx, pr * (1- 2 * rate)))
            elif state == 'percoxyg':
                self.state.percoxyg_state = max(0, self.state.percoxyg_state - 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                self.state.percoxyg_state = min(1, self.state.percoxyg_state + 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                temp.append((idx, pr * (1- 2 * rate)))
            elif state == 'glucose':
                self.state.glucose_state = max(0, self.state.glucose_state - 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                self.state.set_state_by_idx(idx, idx_type='obs', diabetic_idx=0)
                self.state.glucose_state = min(1, self.state.glucose_state + 1)
                temp.append((self.state.get_state_idx(), pr * rate))
                temp.append((idx, pr * (1- 2 * rate)))
        _temp = {}
        for (idx, pr) in temp:
            if idx not in _temp.keys():
                _temp[idx] = pr
            else:
                _temp[idx] += pr

        return _temp.items()

    def transition_fluctuate(self, hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate, \
        glucose_fluctuate, probs):
        '''
        all (non-treatment) states fluctuate +/- 1 w.p. .1
        exception: glucose flucuates +/- 1 w.p. .3 if diabetic
        '''
        # only retain non zero prob state
        temp = []
        for (idx, pr) in enumerate(probs):
            if pr > 0:
                temp.append((idx, pr))
        if hr_fluctuate:
            temp = self._next_transition(temp, 'hr', 0.1)
        if sysbp_fluctuate:
            temp = self._next_transition(temp, 'sysbp', 0.1)
        if percoxyg_fluctuate:
            temp = self._next_transition(temp, 'percoxyg', 0.1)
        if glucose_fluctuate:
            temp = self._next_transition(temp, 'glucose', 0.1)
        return temp
        
        '''
        if hr_fluctuate:
            hr_prob = np.random.uniform(0,1)
            if hr_prob < 0.1:
                self.state.hr_state = max(0, self.state.hr_state - 1)
            elif hr_prob < 0.2:
                self.state.hr_state = min(2, self.state.hr_state + 1)
        
        if sysbp_fluctuate:
            sysbp_prob = np.random.uniform(0,1)
            if sysbp_prob < 0.1:
                self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            elif sysbp_prob < 0.2:
                self.state.sysbp_state = min(2, self.state.sysbp_state + 1)
        
        if percoxyg_fluctuate:
            percoxyg_prob = np.random.uniform(0,1)
            if percoxyg_prob < 0.1:
                self.state.percoxyg_state = max(0, self.state.percoxyg_state - 1)
            elif percoxyg_prob < 0.2:
                self.state.percoxyg_state = min(1, self.state.percoxyg_state + 1)
        
        if glucose_fluctuate:
            glucose_prob = np.random.uniform(0,1)
            if self.state.diabetic_idx == 0:
                if glucose_prob < 0.1:
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < 0.2:
                    self.state.glucose_state = min(1, self.state.glucose_state + 1)
            else:
                if glucose_prob < 0.3:
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < 0.6:
                    self.state.glucose_state = min(4, self.state.glucose_state + 1)
        '''

    def calculateReward(self):
        num_abnormal = self.state.get_num_abnormal()
        if num_abnormal >= 3:
            # print("NOOOO FAILED")
            return -1 # works with -100
        elif num_abnormal == 0 and not self.state.on_treatment():
            # print('YESSSS CALCULATE')
            return 1 # works w/ 100
        return 0

    def transition(self, action, p=1.0):
        cur_reward = self.calculateReward()

        # start with 1.0 prob and for this action, change the probability to the next state
        self.state = self.state.copy_state() 
        probs = np.zeros((720,))
        state_idx = self.state.get_state_idx()
        assert(state_idx < 720)
        probs[state_idx] = 1.0

        if action.antibiotic == 1:
            probs = self.transition_antibiotics_on(probs)
            hr_fluctuate = False
            sysbp_fluctuate = False
        elif self.state.antibiotic_state == 1:
            probs = self.transition_antibiotics_off(probs)
            hr_fluctuate = False
            sysbp_fluctuate = False
        else:
            hr_fluctuate = True
            sysbp_fluctuate = True

        if action.ventilation == 1:
            probs = self.transition_vent_on(probs)
            percoxyg_fluctuate = False
        elif self.state.vent_state == 1:
            probs = self.transition_vent_off(probs)
            percoxyg_fluctuate = False
        else:
            percoxyg_fluctuate = True

        glucose_fluctuate = True

        if action.vasopressors == 1:
            probs = self.transition_vaso_on(probs)
            sysbp_fluctuate = False
            glucose_fluctuate = False
        elif self.state.vaso_state == 1:
            probs = self.transition_vaso_off(probs)
            sysbp_fluctuate = False
        
        # print(state_idx, sum(probs))
        # what if turn off transition fluctuate?
        probs = self.transition_fluctuate(hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate, glucose_fluctuate, probs) 
        _probs = np.zeros((720,))
        # _, y = zip(*probs)
        # denom = sum(y)
        # print(denom)
        for (idx, pr) in probs:
            _probs[int(idx)] = pr # / denom
        # import pdb;pdb.set_trace()
        print(sum(_probs))
        # assert(sum(_probs) == 1.)
        # reward is only given for s (a doesn't matter!)
        # return probs from every state s --> 720 array
        return _probs, cur_reward # self.calculateReward()

    def select_actions(self):
        assert self.policy_array is not None
        probs = self.policy_array[
                    self.state.get_state_idx(self.policy_idx_type)
                ]
        aev_idx = np.random.choice(np.arange(Action.NUM_ACTIONS_TOTAL), p=probs)
        return Action(action_idx = aev_idx)
