     
        ##find the first time the action chosen was not consistent with the policy
        first_break = 0 #-next(i for i,x in enumerate(episode) if target_policy(x[0])[x[1]] == 0)
        for i in range(len(episode))[::-1]:
            observation, state, _ = episode[i]
            if target_policy(observation)[state] == 0:
                first_break = i
                break
            

        for tuplething in enumerate(episode[first_break:]):
            i, (observation, action, reward) = tuplething
            returns_sum[(observation, action)] += reward*discount_factor**(i-1)
            returns_count[(observation, action)] += 1
        
        
            
            
            

        ##Use the updated Q matrix to construct a new greedy target policy
        for observation, action in returns_sum:
            Q[observation][action] = returns_sum[(observation, action)]/returns_count[(observation, action)]

        ##repeat