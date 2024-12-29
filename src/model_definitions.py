# src/model_definitions.py

import torch
import torch.nn as nn

###############################################################################
# (1) SharedMTLModel -- Basic ReLU Output (ver 1)
###############################################################################
class SharedMTLModel_v1(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(SharedMTLModel_v1, self).__init__()
        # Shared layers with Dropout
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Task-specific output layers
        self.task1 = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.task2 = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.task3 = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.task4 = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        out1 = self.task1(shared_representation)
        out2 = self.task2(shared_representation)
        out3 = self.task3(shared_representation)
        out4 = self.task4(shared_representation)
        return out1, out2, out3, out4


###############################################################################
# (2) SharedMTLModel -- Another version with hidden_size1, hidden_size2
###############################################################################
class SharedMTLModel(nn.Module):
    def __init__(self, input_size=5, hidden_size1=128, hidden_size2=64, dropout_rate=0.5):
        super(SharedMTLModel, self).__init__()
        self.num_tasks = 4
        
        # Shared layers with Dropout
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Task-specific output layers
        self.task1 = nn.Sequential(nn.Linear(hidden_size2, 1), nn.ReLU())
        self.task2 = nn.Sequential(nn.Linear(hidden_size2, 1), nn.ReLU())
        self.task3 = nn.Sequential(nn.Linear(hidden_size2, 1), nn.ReLU())
        self.task4 = nn.Sequential(nn.Linear(hidden_size2, 1), nn.ReLU())

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        out1 = self.task1(shared_representation)
        out2 = self.task2(shared_representation)
        out3 = self.task3(shared_representation)
        out4 = self.task4(shared_representation)
        return out1, out2, out3, out4


###############################################################################
# (3) SeparateMTLModel -- ReLU Activation in Output Layers
###############################################################################
class SeparateMTLModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SeparateMTLModel, self).__init__()
        self.num_tasks = 4
        
        # Shared base layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Task-specific branches
        self.task1_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )
        self.task2_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )
        self.task3_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )
        self.task4_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        out1 = self.task1_branch(shared_representation)
        out2 = self.task2_branch(shared_representation)
        out3 = self.task3_branch(shared_representation)
        out4 = self.task4_branch(shared_representation)
        return out1, out2, out3, out4


###############################################################################
# (4) Shared MTL Model with Uncertainty
###############################################################################
class SharedMTLModelWithUncertainty(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5, num_tasks=4):
        super(SharedMTLModelWithUncertainty, self).__init__()
        self.num_tasks = num_tasks
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Task-specific outputs
        self.task_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, 1)) for _ in range(num_tasks)]
        )
        # Learnable log variance
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, x):
        shared_representation = self.shared_layers(x)
        outputs = [layer(shared_representation) for layer in self.task_layers]
        return outputs


###############################################################################
# (5) SeparateMTLModelWithUncertainty
###############################################################################
class SeparateMTLModelWithUncertainty(nn.Module):
    def __init__(self, input_size, hidden_size, num_tasks=4):
        super(SeparateMTLModelWithUncertainty, self).__init__()
        self.num_tasks = num_tasks
        # Shared base layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Task-specific branches
        self.task_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            ) for _ in range(num_tasks)
        ])
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, x):
        shared_representation = self.shared_layers(x)
        outputs = [branch(shared_representation) for branch in self.task_branches]
        return outputs


###############################################################################
# (6) Ref_Based
###############################################################################
class Ref_Based(nn.Module):
    def __init__(self, input_size=5, hidden_size1=128, hidden_size2=64, dropout_rate=0.5, num_tasks=4):
        super(Ref_Based, self).__init__()
        self.num_tasks = num_tasks
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Energy-specific
        self.energy_layers = nn.Sequential(
            nn.Linear(hidden_size2, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
        # Cost-specific
        self.cost_layers = nn.Sequential(
            nn.Linear(hidden_size2, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Emission-specific
        self.emission_layers = nn.Sequential(
            nn.Linear(hidden_size2, 14),
            nn.ReLU(),
            nn.Linear(14, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # Comfort-specific
        self.comfort_layers = nn.Sequential(
            nn.Linear(hidden_size2, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )
        
        # For Uncertainty
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, x):
        shared_representation = self.shared_layers(x)
        energy = self.energy_layers(shared_representation)
        cost = self.cost_layers(shared_representation)
        emission = self.emission_layers(shared_representation)
        comfort = self.comfort_layers(shared_representation)
        return energy, cost, emission, comfort


###############################################################################
# (7) Data_Based
###############################################################################
class Data_Based(nn.Module):
    def __init__(self, input_size=5, hidden_size1=128, hidden_size2=64, 
                 shared_energy_emission_size=32, shared_comfort_size=32, 
                 dropout_rate=0.5, num_tasks=4):
        super(Data_Based, self).__init__()
        self.num_tasks = num_tasks
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Shared for Energy & Emission
        self.shared_energy_emission = nn.Sequential(
            nn.Linear(hidden_size2, shared_energy_emission_size),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.fc_energy = nn.Sequential(nn.Linear(shared_energy_emission_size, 1))
        self.fc_cost = nn.Sequential(
            nn.Linear(hidden_size2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.fc_emission = nn.Sequential(nn.Linear(shared_energy_emission_size, 1))
        self.fc_comfort = nn.Sequential(
            nn.Linear(hidden_size2, shared_comfort_size),
            nn.ReLU(),
            nn.Linear(shared_comfort_size, 1)
        )
        
        # For Uncertainty
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, x):
        shared_representation = self.shared_layers(x)
        
        cost = self.fc_cost(shared_representation)
        
        # Shared for Energy & Emission
        shared_energy_emission = self.shared_energy_emission(shared_representation)
        energy = self.fc_energy(shared_energy_emission)
        emission = self.fc_emission(shared_energy_emission)
        
        # Comfort
        comfort = self.fc_comfort(shared_representation)
        
        return energy, cost, emission, comfort


###############################################################################
# (8) More_Shared
###############################################################################
class More_Shared(nn.Module):
    def __init__(self, input_size=5, hidden_size1=256, hidden_size2=128, hidden_size3=64,
                 dropout_rate=0.5, num_tasks=4):
        super(More_Shared, self).__init__()
        self.num_tasks = num_tasks

        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.energy_layers = nn.Sequential(
            nn.Linear(hidden_size3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.cost_layers = nn.Sequential(
            nn.Linear(hidden_size3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.emission_layers = nn.Sequential(
            nn.Linear(hidden_size3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.comfort_layers = nn.Sequential(
            nn.Linear(hidden_size3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        energy = self.energy_layers(shared_representation)
        cost = self.cost_layers(shared_representation)
        emission = self.emission_layers(shared_representation)
        comfort = self.comfort_layers(shared_representation)
        return energy, cost, emission, comfort


###############################################################################
# (9) Few_Shared
###############################################################################
class Few_Shared(nn.Module):
    def __init__(self, input_size=5, hidden_size_shared=128, dropout_rate=0.5, num_tasks=4):
        super(Few_Shared, self).__init__()
        self.num_tasks = num_tasks

        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size_shared),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.energy_branch = nn.Sequential(
            nn.Linear(hidden_size_shared, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.cost_branch = nn.Sequential(
            nn.Linear(hidden_size_shared, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.emission_branch = nn.Sequential(
            nn.Linear(hidden_size_shared, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.comfort_branch = nn.Sequential(
            nn.Linear(hidden_size_shared, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        energy = self.energy_branch(shared_representation)
        cost = self.cost_branch(shared_representation)
        emission = self.emission_branch(shared_representation)
        comfort = self.comfort_branch(shared_representation)
        return energy, cost, emission, comfort


###############################################################################
# (10) Deep_Balanced
###############################################################################
class Deep_Balanced(nn.Module):
    def __init__(self, input_size=5, hidden_size1=200, hidden_size2=100,
                 hidden_size3=50, dropout_rate=0.5, num_tasks=4):
        super(Deep_Balanced, self).__init__()
        self.num_tasks = num_tasks

        # Shared Layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Energy
        self.energy_layers = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, hidden_size3 // 2),
            nn.ReLU(),
            nn.Linear(hidden_size3 // 2, 1)
        )

        # Cost
        self.cost_layers = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size3 * 2),
            nn.ReLU(),
            nn.Linear(hidden_size3 * 2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, 1)
        )

        # Emission
        emission_output_size = int(hidden_size3 * 1.5)
        self.emission_layers = nn.Sequential(
            nn.Linear(hidden_size2, emission_output_size),
            nn.ReLU(),
            nn.Linear(emission_output_size, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, 1)
        )

        # Comfort
        self.comfort_layers = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size3 // 2),
            nn.ReLU(),
            nn.Linear(hidden_size3 // 2, hidden_size3 // 4),
            nn.ReLU(),
            nn.Linear(hidden_size3 // 4, 1)
        )

        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        energy = self.energy_layers(shared_representation)
        cost = self.cost_layers(shared_representation)
        emission = self.emission_layers(shared_representation)
        comfort = self.comfort_layers(shared_representation)
        return energy, cost, emission, comfort
