import sys
import os
import time
import joblib

# Importa classes
from decision_tree import DecisionTree
from global_resources import GlobalResources
from rescuer import Rescuer
from vs.environment import Env
from explorer import Explorer
from fuzzy import Fuzzy

# Carregar modelos treinados
mlp_regressor = joblib.load('mlp_regressor.pkl')
tree_regressor = joblib.load('tree_regressor.pkl')

# Função para estimar gravidade
def estimate_gravity(vital_signs):
    qPA, pulso, freq_respiratoria = vital_signs
    gravity = tree_regressor.predict([[qPA, pulso, freq_respiratoria]])[0]
    return gravity


def main(data_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    # Instantiate the environment
    env = Env(data_folder)
    
    # Config files for the agents
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")
    explorer_file = os.path.join(data_folder, "explorer_config.txt")
    
    fuzzy = Fuzzy()
    decision_tree = DecisionTree()
    
    # Instantiate agents rescuer and explorer
    resc1 = Rescuer(env, rescuer_file, 0, fuzzy, estimate_gravity)
    resc2 = Rescuer(env, rescuer_file, 1, fuzzy, estimate_gravity)
    resc3 = Rescuer(env, rescuer_file, 2, decision_tree, estimate_gravity)
    resc4 = Rescuer(env, rescuer_file, 3, decision_tree, estimate_gravity)

    # Explorer needs to know rescuer to send the map
    # That's why rescuer is instantiated before
    expl1 = Explorer(env, explorer_file, resc1, 0)
    expl2 = Explorer(env, explorer_file, resc2, 1)
    expl3 = Explorer(env, explorer_file, resc3, 2)
    expl4 = Explorer(env, explorer_file, resc4, 3)
    
    global_resources = GlobalResources()
    
    global_resources.add_rescuer(resc1)\
        .add_rescuer(resc2)\
        .add_rescuer(resc3)\
        .add_rescuer(resc4)\
        .add_explorer(expl1)\
        .add_explorer(expl2)\
        .add_explorer(expl3)\
        .add_explorer(expl4)
        
    expl1.add_global_resources(global_resources)
    expl2.add_global_resources(global_resources)
    expl3.add_global_resources(global_resources)
    expl4.add_global_resources(global_resources)

    # Run the environment simulator
    env.run()
    
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_10v_12X12")
        
    main(data_folder_name)
