# CREATING NEW FEATURES

def newfeature(temp_dict, new_test_dict):
    childs = males = 0
    #--------------train data------------#
    for key in temp_dict:
        # Family Relation
        # Alone marks whether the person onboard is with family or individual
        family = int(temp_dict[key]['Parch']) + int(temp_dict[key]['SibSp']) + 1
        temp_dict[key]['Alone'] = '1' if family == 1 else 0
        # Drop features Parch and Sibling
        del temp_dict[key]['Parch']
        del temp_dict[key]['SibSp']

        # gender
        # Converting Categorical Data into numeric data and splitting gender in male, female and child
        temp_dict[key]['Child'] = '1' if temp_dict[key]['Age'] < 17 else 0
        temp_dict[key]['Female'] = '1' if temp_dict[key]['Sex'] == 'female' else 0
        temp_dict[key]['Male'] = '1' if temp_dict[key]['Sex'] == 'male' else 0
        # Drop feature Sex
        del temp_dict[key]['Sex']

        #Embarked
        # Converting Categorical Data into numeric data
        temp_dict[key]['S'] = '1' if temp_dict[key]['Embarked'] == 'S' else 0
        temp_dict[key]['C'] = '1' if temp_dict[key]['Embarked'] == 'C' else 0
        temp_dict[key]['Q'] = '1' if temp_dict[key]['Embarked'] == 'Q' else 0
        # Drop feature Embarked
        del temp_dict[key]['Embarked']

        #--------------test data-------------#

    for key in new_test_dict:
        # Family Relation
        family = int(new_test_dict[key]['Parch']) + int(new_test_dict[key]['SibSp']) + 1
        new_test_dict[key]['Alone'] = '1' if family == 1 else 0

        del new_test_dict[key]['Parch']
        del new_test_dict[key]['SibSp']

        # gender
            
        new_test_dict[key]['Child'] = '1' if new_test_dict[key]['Age'] < 17 else 0
        new_test_dict[key]['Female'] = '1' if new_test_dict[key]['Sex'] == 'female' else 0
        new_test_dict[key]['Male'] = '1' if new_test_dict[key]['Sex'] == 'male' else 0
        
        del new_test_dict[key]['Sex']

        #Embarked

        new_test_dict[key]['S'] = '1' if new_test_dict[key]['Embarked'] == 'S' else 0
        new_test_dict[key]['C'] = '1' if new_test_dict[key]['Embarked'] == 'C' else 0
        new_test_dict[key]['Q'] = '1' if new_test_dict[key]['Embarked'] == 'Q' else 0
        
        del new_test_dict[key]['Embarked']

    return temp_dict, new_test_dict
