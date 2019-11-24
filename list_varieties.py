import os

def successful_varieties():
    main_path = '/home/viktor/Malware-Analysis-Results/successful-analysis'

    varieties = []
    for family in os.listdir(main_path):
        family_path = f'{main_path}/{family}'
        for variety in os.listdir(family_path):
            if(os.listdir(f'{family_path}/{variety}')):
                varieties.append(f'{family}/{variety}')
    return varieties