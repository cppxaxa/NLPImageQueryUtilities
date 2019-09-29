from allennlp.predictors.predictor import Predictor
from nltk.stem import PorterStemmer

dependencyParserModelPath = "allennlp_models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz"
inputStr = 'A dog is laying in the grass with a frisbee in mouth'

class QuestionParser:
    def __init__(self, dependencyParserModelPath):
        self.dependencyParserModelPath = dependencyParserModelPath
        self.parser = Predictor.from_path(dependencyParserModelPath)
        self.caseHandlers = []

    def addCaseHandlers(self, caseHandlerObject):
        caseHandlerObject.setDependencyParserInstance(self.parser)
        self.caseHandlers.append(caseHandlerObject)
    
    def preprocessInput(self, inputQuestion):
        inputQuestion = inputQuestion.strip()
        inputQuestion = inputQuestion.replace('?', '')
        inputQuestion = inputQuestion.replace('.', '')
        inputQuestion = inputQuestion.replace('@', '')
        inputQuestion = inputQuestion.replace('#', '')
        inputQuestion = inputQuestion.replace('  ', ' ')
        inputQuestion = inputQuestion.lower()

        return inputQuestion

    def generateQueryString(self, inputQuestion):
        inputQuestionPreprocessed = self.preprocessInput(inputQuestion)

        outputList = []
        for handler in self.caseHandlers:
            try:
                res = handler.extractKeywords(inputQuestionPreprocessed)
                if res is not None and res.strip() != '':
                    outputList.append(res)
            except:
                pass
        return ' '.join(outputList)


'''
class CaseHandlerTemplate:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBG':
            children = root['children']
            targetLink = 'prt'
            targetAttr = 'RP'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                keyword = root['word'] + foundItem['word']
                return keyword

        return None

'''

class VBG_PRT_RP_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBG':
            children = root['children']
            targetLink = 'prt'
            targetAttr = 'RP'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                keyword = root['word'] + foundItem['word']
                return keyword

        return None



class VBG_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBG':
            keyword = root['word']
            return keyword

        return None


class VBZ_ADVMOD_RB_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBZ':
            children = root['children']
            targetLink = 'advmod'
            targetAttr = 'RB'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                keyword = root['word'] + foundItem['word']
                return keyword

        return None


class VBZ_EXPL_EX_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBZ':
            children = root['children']
            targetLink = 'expl'
            targetAttr = 'EX'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                keyword = root['word'] + foundItem['word']
                return keyword

        return None



class NN_PARTMOD_VBG_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'NN':
            children = root['children']
            targetLink = 'partmod'
            targetAttr = 'VBG'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                ps = PorterStemmer()
                stemmedNoun = ps.stem(root['word'])
                stemmedVerb = ps.stem(foundItem['word'])
                keyword = stemmedNoun + stemmedVerb
                return keyword

        return None



class VBZ_NSUBJ_NN_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleWord(self, word):
        if word == 'doe':
            return 'do'
        elif word == 'ha':
            return 'has'
        elif word == 'have':
            return 'has'
        elif word == 'had':
            return 'has'

        return word

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBZ':
            children = root['children']
            targetLink = 'nsubj'
            targetAttr = 'NN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                ps = PorterStemmer()
                stemmedVerb = ps.stem(root['word'])
                simpleWord = self.getSimpleWord(stemmedVerb)
                keyword = foundItem['word'] + simpleWord
                return keyword

        return None



class VBZ_PREP_IN_POBJ_NN_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBZ':
            children = root['children']
            targetLink = 'prep'
            targetAttr = 'IN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                children2nd = foundItem['children']
                targetLink2nd = 'pobj'
                targetAttr2nd = 'NN'

                foundItem2nd = None

                for i in range(len(children2nd)):
                    item = children2nd[i]
                    if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                        foundItem2nd = item
                        break
                
                if foundItem2nd is not None:
                    keyword = foundItem['word'] + foundItem2nd['word']

                    return keyword

        return None




class VB_NSUBJ_NN_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleWord(self, word):
        if word == 'doe':
            return 'do'
        elif word == 'ha':
            return 'has'
        elif word == 'have':
            return 'has'
        elif word == 'had':
            return 'has'

        return word

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VB':
            children = root['children']
            targetLink = 'nsubj'
            targetAttr = 'NN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                ps = PorterStemmer()
                stemmedVerb = ps.stem(root['word'])
                simpleWord = self.getSimpleWord(stemmedVerb)
                keyword = foundItem['word'] + simpleWord
                return keyword

        return None



class VBD_NSUBJ_NN_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleWord(self, word):
        if word == 'doe':
            return 'do'
        elif word == 'ha':
            return 'has'
        elif word == 'have':
            return 'has'
        elif word == 'had':
            return 'has'

        return word

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBD':
            children = root['children']
            targetLink = 'nsubj'
            targetAttr = 'NN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                ps = PorterStemmer()
                stemmedVerb = ps.stem(root['word'])
                simpleWord = self.getSimpleWord(stemmedVerb)
                keyword = foundItem['word'] + simpleWord
                return keyword

        return None



class NN_COP_VBZ_ADVMOD_WRB_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def simplifyWord(self, word):
        if word == 'are':
            return 'is'
        return word

    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'NN':
            children = root['children']
            targetLink = 'cop'
            targetAttr = 'VBZ'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                children2nd = foundItem['children']
                targetLink2nd = 'advmod'
                targetAttr2nd = 'WRB'

                foundItem2nd = None

                for i in range(len(children2nd)):
                    item = children2nd[i]
                    if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                        foundItem2nd = item
                        break
                
                if foundItem2nd is not None:
                    ps = PorterStemmer()

                    questionWord = foundItem2nd['word']
                    joiningWord = foundItem['word']
                    nounWord = root['word']

                    joiningWord = self.simplifyWord(joiningWord)
                    nounWord = ps.stem(nounWord)

                    keyword = questionWord + joiningWord + nounWord

                    return keyword

        return None





class VBP_NSUBJ_NNS_AMOD_JJ_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleSynonym(self, word):
        if word == 'many':
            return 'count'
        return word
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBP':
            children = root['children']
            targetLink = 'nsubj'
            targetAttr = 'NNS'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                children2nd = foundItem['children']
                targetLink2nd = 'amod'
                targetAttr2nd = 'JJ'

                foundItem2nd = None

                for i in range(len(children2nd)):
                    item = children2nd[i]
                    if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                        foundItem2nd = item
                        break
                
                if foundItem2nd is not None:
                    ps = PorterStemmer()

                    noun = ps.stem(foundItem['word'])
                    simpleSynonym = self.getSimpleSynonym(foundItem2nd['word'])
                    keyword = noun + simpleSynonym

                    return keyword

        return None




class VBP_NSUBJ_NN_AMOD_JJ_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleSynonym(self, word):
        if word == 'many':
            return 'count'
        return word
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBP':
            children = root['children']
            targetLink = 'nsubj'
            targetAttr = 'NN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                children2nd = foundItem['children']
                targetLink2nd = 'amod'
                targetAttr2nd = 'JJ'

                foundItem2nd = None

                for i in range(len(children2nd)):
                    item = children2nd[i]
                    if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                        foundItem2nd = item
                        break
                
                if foundItem2nd is not None:
                    ps = PorterStemmer()

                    noun = ps.stem(foundItem['word'])
                    simpleSynonym = self.getSimpleSynonym(foundItem2nd['word'])
                    keyword = noun + simpleSynonym

                    return keyword

        return None




class NN_PREP_IN_POBJ_NN_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleSynonym(self, word):
        if word == 'many':
            return 'count'
        return word
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'NN':
            children = root['children']
            targetLink = 'prep'
            targetAttr = 'IN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                children2nd = foundItem['children']
                targetLink2nd = 'pobj'
                targetAttr2nd = 'NN'

                foundItem2nd = None

                for i in range(len(children2nd)):
                    item = children2nd[i]
                    if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                        foundItem2nd = item
                        break
                
                if foundItem2nd is not None:
                    ps = PorterStemmer()

                    noun = ps.stem(root['word'])
                    simpleSynonym = self.getSimpleSynonym(foundItem2nd['word'])
                    keyword = simpleSynonym + noun

                    return keyword

        return None




class WP_NSUBJ_NN_PREP_NN_PREP_IN_POBJ_NN_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleSynonym(self, word):
        if word == 'many':
            return 'count'
        return word
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'WP':
            children = root['children']
            targetLink = 'nsubj'
            targetAttr = 'NN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                children2nd = foundItem['children']
                targetLink2nd = 'prep'
                targetAttr2nd = 'IN'

                foundItem2nd = None

                for i in range(len(children2nd)):
                    item = children2nd[i]
                    if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                        foundItem2nd = item
                        break
                
                if foundItem2nd is not None:
                    children3rd = foundItem2nd['children']
                    targetLink3rd = 'pobj'
                    targetAttr3rd = 'NN'

                    foundItem3rd = None

                    for i in range(len(children3rd)):
                        item = children3rd[i]
                        if item['link'].lower() == targetLink3rd and item['attributes'][0] == targetAttr3rd:
                            foundItem3rd = item
                            break
                    
                    if foundItem3rd is not None:
                        ps = PorterStemmer()

                        noun = ps.stem(foundItem3rd['word'])
                        simpleSynonym = self.getSimpleSynonym(foundItem['word'])
                        keyword = noun + simpleSynonym

                        return keyword

        return None





class NN_PREP_IN_POBJ_NNS_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleSynonym(self, word):
        if word == 'many':
            return 'count'
        return word
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'NN':
            children = root['children']
            targetLink = 'prep'
            targetAttr = 'IN'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                children2nd = foundItem['children']
                targetLink2nd = 'pobj'
                targetAttr2nd = 'NNS'

                foundItem2nd = None

                for i in range(len(children2nd)):
                    item = children2nd[i]
                    if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                        foundItem2nd = item
                        break
                
                if foundItem2nd is not None:
                    ps = PorterStemmer()

                    noun = ps.stem(foundItem2nd['word'])
                    simpleSynonym = self.getSimpleSynonym(root['word'])
                    keyword = noun + simpleSynonym

                    return keyword

        return None




class NN_ADVMOD_WRB_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'NN':
            children = root['children']
            targetLink = 'advmod'
            targetAttr = 'WRB'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            if foundItem is not None:
                if foundItem['word'].strip().lower() == 'where':
                    keyword = 'whereis' + root['word']
                    return keyword

        return None



class VBP_NSUBJ_NN_OR_NNS_VBP_ADVMOD_WRB_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance

    def getSimpleSynonym(self, word):
        if word == 'many':
            return 'count'
        return word
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'VBP':
            children = root['children']
            targetLink = 'nsubj'
            targetAttr = 'NN'
            targetAttrAlter = 'NNS'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink:
                    if item['attributes'][0] == targetAttr or item['attributes'][0] == targetAttrAlter:
                        foundItem = item
                        break
            
            children2nd = root['children']
            targetLink2nd = 'advmod'
            targetAttr2nd = 'WRB'

            foundItem2nd = None

            for i in range(len(children2nd)):
                item = children2nd[i]
                if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                    foundItem2nd = item
                    break
            
            if foundItem is not None and foundItem2nd is not None:
                if foundItem2nd['word'] == 'where':
                    ps = PorterStemmer()
                    nounWord = ps.stem(foundItem['word'])
                    keyword = 'whereis' + nounWord
                    return keyword

        return None




class NNS_COP_VBP_ADVMOD_WRB_Class:
    def setDependencyParserInstance(self, dependencyParserInstance):
        self.dependencyParserInstance = dependencyParserInstance
    
    def extractKeywords(self, inputQuestion):
        res = self.dependencyParserInstance.predict(sentence=inputQuestion)

        root = res['hierplane_tree']['root']
        attr = root['attributes'][0]
        if attr == 'NNS':
            children = root['children']
            targetLink = 'cop'
            targetAttr = 'VBP'
            foundItem = None

            for i in range(len(children)):
                item = children[i]
                if item['link'].lower() == targetLink and item['attributes'][0] == targetAttr:
                    foundItem = item
                    break
            
            children2nd = foundItem['children']
            targetLink2nd = 'advmod'
            targetAttr2nd = 'WRB'

            foundItem2nd = None

            for i in range(len(children2nd)):
                item = children2nd[i]
                if item['link'].lower() == targetLink2nd and item['attributes'][0] == targetAttr2nd:
                    foundItem2nd = item
                    break
            
            if foundItem is not None and foundItem2nd is not None:
                if foundItem2nd['word'] == 'where':
                    ps = PorterStemmer()
                    nounWord = ps.stem(root['word'])
                    keyword = 'whereis' + nounWord
                    return keyword

        return None



class QuestionParserFactory:
    @staticmethod
    def GetQuestionParser():
        VBG_PRT_RP = VBG_PRT_RP_Class()
        VBG = VBG_Class()
        VBZ_ADVMOD_RB = VBZ_ADVMOD_RB_Class()
        VBZ_EXPL_EX = VBZ_EXPL_EX_Class()
        NN_PARTMOD_VBG = NN_PARTMOD_VBG_Class()
        VBZ_NSUBJ_NN = VBZ_NSUBJ_NN_Class()
        VBZ_PREP_IN_POBJ_NN = VBZ_PREP_IN_POBJ_NN_Class()
        VB_NSUBJ_NN = VB_NSUBJ_NN_Class()
        VBD_NSUBJ_NN = VBD_NSUBJ_NN_Class()
        NN_COP_VBZ_ADVMOD_WRB = NN_COP_VBZ_ADVMOD_WRB_Class()
        VBP_NSUBJ_NNS_AMOD_JJ = VBP_NSUBJ_NNS_AMOD_JJ_Class()
        VBP_NSUBJ_NN_AMOD_JJ = VBP_NSUBJ_NN_AMOD_JJ_Class()
        NN_PREP_IN_POBJ_NN = NN_PREP_IN_POBJ_NN_Class()
        WP_NSUBJ_NN_PREP_NN_PREP_IN_POBJ_NN = WP_NSUBJ_NN_PREP_NN_PREP_IN_POBJ_NN_Class()
        NN_PREP_IN_POBJ_NNS = NN_PREP_IN_POBJ_NNS_Class()
        NN_ADVMOD_WRB = NN_ADVMOD_WRB_Class()
        VBP_NSUBJ_NN_OR_NNS_VBP_ADVMOD_WRB = VBP_NSUBJ_NN_OR_NNS_VBP_ADVMOD_WRB_Class()
        NNS_COP_VBP_ADVMOD_WRB = NNS_COP_VBP_ADVMOD_WRB_Class()



        questionParser = QuestionParser(dependencyParserModelPath)
        questionParser.addCaseHandlers(VBG_PRT_RP)
        questionParser.addCaseHandlers(VBG)
        questionParser.addCaseHandlers(VBZ_ADVMOD_RB)
        questionParser.addCaseHandlers(VBZ_EXPL_EX)
        questionParser.addCaseHandlers(NN_PARTMOD_VBG)
        questionParser.addCaseHandlers(VBZ_NSUBJ_NN)
        questionParser.addCaseHandlers(VBZ_PREP_IN_POBJ_NN)
        questionParser.addCaseHandlers(VB_NSUBJ_NN)
        questionParser.addCaseHandlers(VBD_NSUBJ_NN)
        questionParser.addCaseHandlers(NN_COP_VBZ_ADVMOD_WRB)
        questionParser.addCaseHandlers(VBP_NSUBJ_NNS_AMOD_JJ)
        questionParser.addCaseHandlers(VBP_NSUBJ_NN_AMOD_JJ)
        questionParser.addCaseHandlers(NN_PREP_IN_POBJ_NN)
        questionParser.addCaseHandlers(WP_NSUBJ_NN_PREP_NN_PREP_IN_POBJ_NN)
        questionParser.addCaseHandlers(NN_PREP_IN_POBJ_NNS)
        questionParser.addCaseHandlers(NN_ADVMOD_WRB)
        questionParser.addCaseHandlers(VBP_NSUBJ_NN_OR_NNS_VBP_ADVMOD_WRB)
        questionParser.addCaseHandlers(NNS_COP_VBP_ADVMOD_WRB)

        return questionParser





if __name__ == '__main__':
    questionParser = QuestionParserFactory.GetQuestionParser()

    questionList = [
        'What is going on?',
        'What is happenning?',
        'Who is there?',
        'What is there?',
        'What is the dog doing?',
        'What the dog does?',
        'What is dog doing?',
        'Where the dog laying?',
        'What is with the dog?',
        'What is with dog?',
        'What the dog has?',
        'What the dog have?',
        'What the dog had?',
        'Where is dog?',
        'How many dogs are there?',
        'How many dog are there?',
        'What is the count of dog?',
        'What is the count of dogs?',
        'Where is the dog?',
        'Where are dog?',
        'Where are dogs?',
        'Where are the dogs?'
    ]

    for question in questionList:
        queryString = questionParser.generateQueryString(question)
        print(queryString)

    # queryString = questionParser.generateQueryString('What the dog had?')
    # print(queryString)

    print('Done')
