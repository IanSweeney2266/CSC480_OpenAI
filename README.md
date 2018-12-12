CSC480 - Open AI Gym Project

    Team: Solo-Dolo
    Author: Ian Sweeney
    Date: 12-11-18

Installation:
    
    testy_gym is the only file required to run my algorithms.
    The only module that must be installed is the gym module.

Operation:
    
    NOTE - The algorithm can be used for any of the discrete control
    environments although I can only guarantee successful training in the
    CartPole and MountainClimber environments.

    To run, change the environment variable in main to the id of the desired
    environment. The bot will then be trained. After training is complete 20
    episodes will be shown with rendering on and the average score will be
    printed. This score along with the name of the environment and the weight
    matrix will be saved to the training cache file in the following format:
        EnvironmentName: average_score
        [weight_matrix]

    To see the output of a bot with pre-trained weights follow the instructions
    in main. All of the pre-trained weight matrices were created with a memory
    size of 1 (defined at the top of the file). To generate a bot with a
    different memory size (not recommended) the training function must be used.
