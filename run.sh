# trpo
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/trpo_Pong-ramDeterministic-v4-2/trpo_Pong-ramDeterministic-v4-2_s864 --itr 2155
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/trpo_Pong-ramDeterministic-v4-1/trpo_Pong-ramDeterministic-v4-1_s737 --itr 1342
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/trpo_Pong-ramDeterministic-v4/trpo_Pong-ramDeterministic-v4_s363 --itr 1851

#ppo
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/ppo_Pong-ramDeterministic-v4-1/ppo_Pong-ramDeterministic-v4-1_s0 --itr 2170
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/ppo_Pong-ramDeterministic-v4-2/ppo_Pong-ramDeterministic-v4-2_s477 --itr 1080
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/ppo_Pong-ramDeterministic-v4-3/ppo_Pong-ramDeterministic-v4-3_s777 --itr 2200

# #dqn
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/dqn_Pong-ramDeterministic-v4/dqn_Pong-ramDeterministic-v4_s746 --itr 2176
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/dqn_Pong-ramDeterministic-v4-1/dqn_Pong-ramDeterministic-v4-1_s365 --itr 2007
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/dqn_Pong-ramDeterministic-v4-2/dqn_Pong-ramDeterministic-v4-2_s867 --itr 1640

# #sacd
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/sacd_Pong-ramDeterministic-v4/sacd_Pong-ramDeterministic-v4_s163 --itr 54
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/sacd_Pong-ramDeterministic-v4-1/sacd_Pong-ramDeterministic-v4-1_s297 --itr 63
# python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/sacd_Pong-ramDeterministic-v4-2/sacd_Pong-ramDeterministic-v4-2_s837 --itr 60

for i in {1..500}
do 
    python -m spinup.run test_policy /Users/khuongle/Documents/4th-semester/cs492/spinningup_stable/spinningup/data/sacd_Breakout-ramDeterministic-v4/sacd_Breakout-ramDeterministic-v4_s923 -n 10 --itr $i
done