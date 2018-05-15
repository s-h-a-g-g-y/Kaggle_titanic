#FEATURE FORMAT

import numpy as np

def featureFormat( dictionary, features, remove_NaN=True):
    return_list = []
    keys = sorted(dictionary.keys())

    for key in keys:
        tmp_list = []
        count = 0
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            try:
                if value=="NaN" and remove_NaN:
                    value = 0
                value = float(value)
            except ValueError as e:
                count+=1
            tmp_list.append( value )


        if features[0] == 'Survived':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list

        append = True
        '''
        if 0 in test_list or "NaN" in test_list:
                append = False
        '''
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features

def testtargetFeatureSplit( data ):

    target = []
    for item in data:
        target.append( item )
        #features.append( item[1:] )

    return target
