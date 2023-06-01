from django.http import HttpResponse
from django.shortcuts import render
from datetime import datetime
from django.shortcuts import redirect


import re
import random
from math import log


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




def about(request):
    return render(request, 'about.html')

def contact(request):
    return HttpResponse("Contact done!")



def registration(request):
    
    cur_time= datetime.now()

     # Read data from CSV file and store in list

 
    
   # Sample data for training and testing
    data = [
        ['do not like', 'negative'],
        ['have good', 'positive'],
        ['do not have good', 'negative'],
        ['do not have bad', 'positive'],
        ['not impressed', 'negative'],
        ['do nice', 'positive'],
        ['It is improving', 'positive'],
        ['more theoritical but less practical', 'negative'],
        ['Nepal education system have both positive & negative side.', 'neutral'],
        ['Nepal education system is  like average.', 'neutral'],
        ['education system is bad.', 'negative'],
        ['good.', 'positive'],
        ['I do not like @Nepal Education system.', 'negative'],
        ['education system', 'neutral'],
        ['Nepal Education system is wonderful.', 'positive'],
        ['very wonderful.', 'positive'],
        ['very happy.', 'positive'],
        ['happy.', 'positive'],
        ['Nepal has a good Education system.', 'positive'],
        ['I am happy with Nepali education system', 'positive'],
        ['Nepal education system can only product skillless labour for forign country', 'negative'],
        ['Nepal education system is not so, bad.', 'neutral'],
        ['it is normal.', 'neutral'],
        ['Nepal is improving in education system.', 'positive'],
        ['I hate Nepal education system because its only collect fees from student.', 'negative'],
        ['I hope in the coming days Nepal education will get more improvement.', 'neutral'],
        ['The quality of education in Nepal needs to be improved.', 'positive'],
        ['The teachers in Nepal are doing a great job despite the challenges they face.', 'positive'],
        ['I am not satisfied with the way exams are conducted in Nepal.', 'negative'],
        ['The curriculum in Nepal needs to be updated to meet the demands of the modern world.', 'positive'],
        ['I really appreciate the efforts of the government to improve the education system in Nepal.', 'positive'],
        ['The infrastructure in many schools and colleges in Nepal is in dire need of improvement.', 'negative'],
        ['I think the education system in Nepal is too focused on memorization and not enough on critical thinking.', 'negative'],
        ['I am glad that more and more students in Nepal are getting access to quality education.', 'positive'],
        ['There is a lack of resources in many schools and colleges in Nepal.', 'negative'],
        ['I think the government should invest more in vocational education in Nepal.', 'neutral'],
        ['I am impressed with the dedication of many Nepali students towards their studies.', 'positive'],
        ['The education system in Nepal needs to be more inclusive of marginalized communities.', 'negative'],
        ['The current state of education in Nepal is unacceptable and needs immediate attention.', 'negative'],
        ['I feel that the education system in Nepal is too exam-oriented and does not focus on overall development of the student.', 'negative'],
        ['I believe that Nepali teachers need more training to be able to provide better education to their students.', 'positive'],
        ['The lack of proper facilities in many schools and colleges in Nepal is a major obstacle to quality education.', 'negative'],
        ['I think the government should make education more affordable for students from low-income families in Nepal.', 'positive'],
        ['The language barrier is a major issue for many students in Nepal, especially in rural areas.', 'neutral'],
        ['I think there should be more emphasis on practical learning in the Nepali education system.', 'neutral'],
        ['The lack of qualified teachers in many schools and colleges in Nepal is a serious problem.', 'negative'],
        ['I believe that technology can play a major role in improving the education system in Nepal.', 'positive'],
        ['The government needs to provide more scholarships to deserving students in Nepal.', 'neutral'],
        ['I am concerned about the high dropout rates in many schools and colleges in Nepal.', 'positive'],
        ['I think there should be more extracurricular activities in Nepali schools to promote overall development of students.', 'neutral'],
        ['I believe that Nepali students have the potential to compete with students from around the world if given proper education and resources.', 'positive'],
        ['The education system in Nepal needs to be more flexible and adaptable to the changing needs of students.', 'positive'],
        ['I am impressed with the resilience of many Nepali students who continue to study despite difficult circumstances.', 'positive'],
        ['The lack of gender equality in the Nepali education system is a major concern.', 'negative'],
        ['I think there should be more vocational training programs for students who are not interested in traditional academic subjects.', 'negative'],
        ['The government should provide more funding to schools and colleges in Nepal to improve the quality of education.', 'positive'],
        ['I believe that community involvement can play a major role in improving the education system in Nepal.', 'positive'],
        ['The lack of proper textbooks and learning materials is a major hindrance to quality education in many schools and colleges in Nepal.', 'negative'],
        ['I think there should be more emphasis on practical skills such as computer literacy and communication skills in the Nepali education system.', 'positive'],
        ['Nepal education system is comfortable', 'positive'],
        ['I like', 'positive'],
        ['We do not like', 'negative'],
        ['I have good view', 'positive'],
        ['I do not have good', 'negative'],
        ['Our system not bad', 'neutral'],
        ['It is dangerous', 'negative'],
        ['It is not impressed', 'negative'],
        ['do nice for the project', 'positive'],
        ['This is bad', 'negative'],
        ['It  is good.', 'positive'],
        ['Everything is good but we need more practice knowledge then theory education.', 'positive'],
        ['Education system in Nepal is being worst due to political student union, no exam in time,no result in time, lack of job oriented education etc.', 'negative'],
        ['Lack of training.', 'negative'],
        ['In my opinion, the educating system in Nepal is primarily based upon theoretical knowledge rather than digital learning. Students in Nepal spend their whole day learning from teachers who teach very bit from books. While in other countries teachers come prepared with teaching and learning materials for their students.', 'negative'],
        ['Overrated, overpriced, pointless. Dissatisfied with present condition.', 'negative'],
        ['This education system sucks thats it.', 'negative'],
        ['YGUYGU.', 'neutral'],
        ['School club activities like sports and different experiments should be prioritized as equal as study to all over the country.', 'positive'],
        ['Nepal education system is outdated and not progressive.', 'negative'],
        ['Nepal Government having slightest amount of effort towards the Education System I think, same old curriculum till date.', 'neutral'],
        ['It is based on theoretical rather than practical which is not good for the student of.', 'negative'],
        ['Since last 5 years the education system of nepal is improving. Many government school as well as private school provides good facility to the student in both academic as well as extra activities.', 'positive'],
        ['Nepal has theoritical education being practised moments, practical education must be prioratized. In some education instituition, they take a lot of fees and provide less facilities to students. Strict laws and rules must be maintained and each education instituition must be check.', 'negative'],
        ['I love Nepal Education system.', 'positive'],
        ['Nepal Education system needs some improvement.', 'neutral'],
        ['Nepal education system is quite good as well as quite bad.', 'neutral'],
        ['Nepal education system is not practical.', 'negative'],
        ['There should be practical in Nepal education system.', 'positive'],
        ['Practical examination should be prioritized in Nepal education system.', 'positive'],
        ['Nepal education system should do survey on satisfaction of student on education system.', 'positive'],
        ['Nepal education system should focus on knowlwdge not on theory.', 'negative'],
        ['Extra curriculam activities should be focus to make student happy.', 'positive'],
        ['Nepal education system is good.', 'positive'],
        ['Nepal education system is bad.', 'negative'],
        ['Nepal education system is updating.', 'positive'],
        ['I am satisfied with Nepal education system', 'positive'],
        ['I love Nepal education system', 'positive'],
        ['Nepal education system needs some  improvement', 'neutral'],
        ['Most of the student are frstated with current education system', 'negative'],
        ['Student are disagreed with the scenario of Nepal education system', 'negative'],
        ['High skilled manpower are lacked to manage Nepal education system', 'negative'],
        ['Nepal education system is somehow effective', 'neutral'],
        ['Nepal education system is acceptable', 'positive'],
        ['Nepal education system is best for theoritical concept', 'positive'],
        ['Nepal education system is unaccepted by foreign platform due to lack of practise', 'negative'],
        ['Student feel difficult to familiar with Nepal education system', 'negative'],
        ['Nepal education system focus on general knowledge of the topic', 'positive'],
        ['Student are satisfied with the friendly nature of teacher', 'positive'],
        ['There are some technologies which need to be implemented for the improvement of Nepal education system', 'neutral'],
        ['Student are coming from different background so they feel difficult to learn', 'neutral'],
        ['There is fair competition in our education system', 'positive'],
        ['Nepal education system is quite pressurized', 'neutral'],
        ['Teacher are skilled so it makes good to understand the topic', 'positive'],
        ['good for improvement', 'positive'],
        ['Most of the student are familiar with Nepal education system based on theriotical concept', 'positive'],
        ['The way of teaching the student is traditional', 'neutral'],
        ['Student are satisfied with the current situation of Nepal education system', 'positive'],
        ['Nepal education system is progressing', 'positive'],
        ['The way of teaching the student is not good', 'neutral'],
        ['Many people are happy with current education system', 'positive'],
        ['Nepal education system does not focus on creativity and interest of student', 'neutral'],
        ['Many student like Nepal education system', 'positive'],
        ['Most of the colleges and school should focused on creativity and growth', 'positive'],
        ['I love Nepal education system', 'positive'],
        ['The way of teaching the student is not so good', 'neutral'],
        ['Nepal education system is fantastic', 'positive'],
        ['Student are happy with the recent education system in Nepal', 'positive'],
        ['Nepal education system is improving and focusing on practical knowledge', 'positive'],
        ['If result is published in right time then it will be great for Nepal education system', 'positive'],
        ['The way of getting the idea from the teacher is impressive', 'positive'],
        ['Student are familiar with the theoritical concept of the study', 'positive'],
        ['Many student gets surprised in exam', 'neutral'],
        ['The education system of Nepal is good as compared to other developing country', 'positive'],
        ['General knowledge is necessary which is not good for specialization', 'neutral'],
        ['The way of teaching is worst', 'negative'],
        ['Nepal education system is not worst', 'positive'],
        ['Nepal education system is effective for general concepts', 'positive'],
        ['not worst', 'positive'],
        ['It is dangerous', 'negative'],
        ['It is worst', 'negative'],
        ['It is not suitable for specialization of any subject', 'negative'],
        ['I think there is political interferrence in Nepal education system', 'neutral'],
        ['Monopoly is main cause for not progressing Nepal education system', 'neutral'],
        ['Many individual are satisfied to get the information from education institute', 'positive'],
        ['There is corruption in all the faculty of education system', 'negative'],
        ['The cost of the facility of getting knowledge is expensive', 'negative'],
        ['I agreed with the present scenario of getting knowledge', 'positive'],
        ['I think it is expensive', 'negative'],
        ['There is expert professor and teacher to guide the student', 'positive'],
        ['Overall system is impressive', 'positive'],
        ['Financially, it is difficult to achieve the education system', 'negative'],
        ['There should be transparency', 'neutral'],
        ['The ranking of Nepal education system is good', 'positive'],
        ['Private colleges are providing better education', 'positive'],
        ['It is safe', 'positive'],
        ['The main reason of not updating our education system is due to corruption', 'neutral'],
        ['Nepal education system should be independent so that it will upgrade smoothly', 'positive'],
        ['Student are facing a lot of problems', 'negative'],
        ['I like Nepal education system', 'positive'],
        ['The technique of learning amazes us', 'positive'],
        ['There is proper punishment for violating the rules', 'positive'],
        ['We cannot get enough knowledge and ideas even paying hign amount of budget', 'negative'],
        ['There is no proper lab and practise matirial in most of the colleges', 'negative'],
        ['Student become more consious towards gaining hign grades', 'positive'],
        ['There  is workforce diversity', 'negative'],
        ['Teacher do not pay attention towards creativity of student', 'negative'],
        ['Student are always in pressure', 'neutral'],
        ['I do not why most of the student are attracted towards Nepal education system', 'positive'],
        ['The graduated student are qualified for higner studies', 'positive'],
        ['We are taught to get the general knowledge of the overall topic', 'positive'],
        ['It helps to provide great platform for the student in Nepal', 'positive'],
        ['Most of the student gets benefits from this system', 'positive'],
        ['Average level of ideas will be provided by Nepal education system', 'positive'],
        ['Nowadays, many youth are disinterested with the current situation of system', 'neutral'],
        ['There should be smooth operation of each administration of the system', 'positive'],
        ['There is huge difference between the expectation and reality before and after joining the education system', 'negative'],
        ['There shound be proper examination result schedule so that student will be satisfied', 'positive'],
        ['There is a lot of corruption in this field', 'negative'],
        ['Student are badly obsrved by the teacher', 'negative'],
        ['There is fair competition in examination', 'positive'],
        ['The syllabus will be completed at the right time', 'positive'],
        ['It ignores the student views', 'negative'],
        ['Our education system is adaptive', 'positive'],
        ['It is totally awesome', 'positive'],
        ['Many student are living in fear with this condition', 'negative'],
        ['It is disappointed that educated student are unemployed', 'negative'],
        ['It helps to uplift the standard of student with the skills', 'positive'],
        ['It is the best version for the theoritical knowledge', 'positive'],
        ['There is dirty mindset people who misusing the power and authority', 'negative'],
        ['It is charming', 'positive'],
        ['There is friendly relationship between the student and teacher', 'positive'],
        ['It is one of the best platform for the communication and sharing ideas', 'positive'],
        ['Overall it provides devine service', 'positive'],
        ['Most of the cases, it ignores the view of student', 'negative'],
        ['It is terrifying for the well managed student', 'negative'],
        ['I enjoy the Nepal education system', 'positive'],
        ['It is excellent because it provides the education services at the same time to all  student', 'positive'],
        ['It is old', 'negative'],
        ['Most of the student considered as favorite', 'positive'],
        ['Many people are unhappy', 'negative'],
        ['Most of the student are motivated by themselves', 'positive'],
        ['Many facilities are missing', 'negative'],
        ['It is playful', 'positive'],
        ['I am happy with the current situation', 'positive'],
        ['Student are sincere to the administration', 'positive'],
        ['The education system id trustworthy', 'positive'],
        ['It is adaptive in nature', 'positive'],
        ['It is attractive to most of the people', 'positive'],
        ['Higher authorities are unconcern about the improvement', 'neutral'],
        ['It is best in quality and education', 'positive'],
        ['It is moderate because of lack of improvement', 'neutral'],
        ['Excellent concepts can be gain about the theory is achieved througn Nepal education system', 'positive'],
        ['Inactive department are there which lacks punctuality', 'neutral'],
        ['It  is neither good not bad', 'neutral'],
        ['Political influence is more in government educational institution', 'negative'],
        ['Every stuent have their own favorite subjects in school', 'positive'],
        ['There is friendly environment in the Nepal education system', 'positive'],
        ['Most of the people are not ready for life after college', 'negative'],
        ['It never focused on passion of student', 'negative'],
        ['Most of the student are come from different background so they fell difficult to learn from same style', 'negative'],
        ['Student become independent to learn new ideas', 'positive'],
        ['It is not creative in nature', 'negative'],
        ['Many student are enjoying the trend of Nepal education system', 'positive'],
        ['Our education is greatest in managing the high number of student in the same field', 'positive'],
        ['After receiving the graduation of Nepal education system, it  is honorable', 'positive'],
        ['It is somehow quite impressive', 'neutral'],
        ['Self dependent manpower were created by the scenatio of current trends', 'positive'],
        ['It is lovely for those syudentwho want to gain only theory knowledge', 'positive'],
        ['Nepali education system is good.', 'positive'],
        ['There is no qualified teacher for teaching', 'negative'],
        ['There is freedom for student to learn anything', 'positive'],
        ['I am biggest supporter of Nepal education because of my self growth and innovation', 'positive'],
        ['There is a big scam in this sector', 'negative'],
        ['New youth are expecting new things from education sectors', 'positive'],
        ['Many student are agreed and getting higher education in other country due to standard level of Nepal education system', 'positive'],
        ['I am familiar with the current situation', 'positive'],
        ['It is not bad', 'neutral'],
        ['It is not good', 'neutral'],
        ['It provides dissatisfaction to student', 'negative'],
        ['The cost is cheap in government colleges', 'positive'],
        ['There is healthy environment with multiple courses in Nepal education system', 'positive'],
        ['Comparing the previous history, it is better now', 'positive'],
        ['Political parties interfaces the growth of education field', 'negative'],
        ['It is getting worst due to the political parties', 'negative'],
        ['I think that the government should make it independent from the political parties', 'positive'],
        ['I believe that near future, it is going to be totally practical', 'positive'],
        ['Many teachers are misusing their power and authorities', 'negative'],
        ['Student should focus only in one particular dream', 'positive'],
        ['It going to be awesome one day', 'positive'],
        ['With the current education system, we can get handsome salary', 'positive'],
        ['Since last 5 years the education system of nepal is improving. Many government school as well as private school provides good facility to the student in both academic as well as extra activities.', 'positive'],
        ['Nepal has theoritical education being practised moments, practical education must be prioratized. In some education instituition, they take a lot of fees and provide less facilities to students. Strict laws and rules must be maintained and each education instituition must be check.', 'neutral'],
        ['Our education system is corrupted and needs to be improvised in syllabus as well as institutions', 'negative'],
        ['I think, same old curriculam till date', 'negative'],
        ['It is not practical which is not good for student', 'negative'],
        ['It is fully centralized which is controlled by senior authorities', 'neutral'],
        ['Education system must be made decentralized so that there will be punctuality in system', 'positive'],
        ['Corrupted authorities should be strictly punished', 'positive'],
        ['Many talented student has to sacrifise their dream in serch of education system', 'negative'],
        ['It is totally based on written exam, it should be done on the basis of practical point of view', 'neutral'],
        ['Student forgot what they have done after the exam', 'negative'],
        ['Government should make policy to provide scholorship to the deserved student fairly', 'positive'],
        ['Student graduated from Nepal education system are fully eligible to study anything in all around the world', 'positive'],
        ['There is no fair fees anong the student of same class.', 'negative'],
        ['Due to broker, there is money minded business in this sector', 'negative'],
        ['The education system of country is popular in term of theoriotical concept', 'positive'],
        ['Private institution should not take high fee from the student', 'neutral'],
        ['I think there is no proper provision of satisfaction to both student and teacher', 'negative'],
        ['New syllabus with new courses are introducing rapidly', 'positive'],
        ['Many practical based technologies and courses are available for student to get the degree', 'positive'],
        ['Top level authorities wants monopoly in Nepal education system', 'neutral'],
        ['Nowadays, many affordable resources are easily available for student to increase their ideas', 'positive'],
        ['As per survey, student are satisifed with the education trend', 'positive'],
        ['The fees of the courses are expensive than aspected', 'negative'],
        ['Colleges are concerning towards extra activites with knowledge', 'positive'],
        ['Several workshop and campaign are organizing to increase the creativity of student', 'positive'],
        ['The management team is handeling the resourses and student properly', 'positive'],
        ['There is no production of skilled manpower', 'negative'],
        ['It is somehow considerable', 'neutral'],
        ['Most of the student who gained higher education in all around the world have base knowledge through Nepal education system', 'positive'],
        ['It is getting popular nowadays', 'positive'],
        ['The government shold make the policy to attract the student from foreigh countries', 'positive'],
        ['It attracts most of the student who want to gain high percentage in exam', 'positive'],
        ['The popularity of the educaton system is gradually increasing', 'positive'],
        ['There is unsatifaction among the student', 'negative'],
        ['There is no fair trade between the fees and knowledge', 'negative'],
        ['Student can select the electives as per their interest in higher studies', 'positive'],
        ['Student has to sacrifise their passion and dreams', 'negative'],
        ['It is not impressive', 'negative'],
        ['I think that all the colleges should be strictly checked during the class time', 'positive'],
        ['I am disagreed', 'neutral'],
        ['It provides the best theoriotical knowledge', 'positive'],
        ['As I can see that there is no discrimination between the student', 'positive'],
        ['The main challenging for student is to read for exam rather than understand', 'negative'],
        ['Reading and writing is the main theme of Nepal education system', 'positive'],
        ['The syllabus is updating regularly which is good for student', 'positive'],
        ['Recently, education system is shifting towards modernization', 'positive'],
        ['In my view, it is providing quality information to student who are interested', 'positive'],
        ['We want some more changes in this field', 'neutral'],
        ['All the staffs should be involved in decision making', 'positive'],
        ['There should be transparency', 'positive'],
        ['The way of treating student is excellent', 'positive'],
        ['The result is outstanding', 'positive'],
        ['We are expecting more than that', 'positive'],
        ['The high level authorities are not working hard to change this system', 'negative'],
        ['It provides quality education', 'positive'],
        ['There is no transparency in fees structure', 'negative'],
        ['It is compulsary to punish those person who affect the growth of Nepal education system', 'positive'],
        ['The lab department needs some improvement', 'positive'],
        ['The student can study and work parallelly', 'positive'],
        ['Student can gain more general knowledge about everything', 'positive'],
        ['I think we should focus on other activities along with knowledge', 'positive'],
        ['I like the way of get satisfaction from Nepal education system', 'positive'],
        ['The private institution are taking more budget from student', 'negative'],
        ['Infrastructure are not well managed by the authorities', 'negative'],
        ['The best part of Nepal education system is equality', 'positive'],
        ['It is satisfaction to the learner', 'positive'],
        ['We cannot get extra knowledge beyond the topic', 'negative'],
        ['We are expecting more in the near future', 'positive'],
        ['In the current situation, student have a lot of options to choose their best subject', 'positive'],
        ['It is very slow', 'negative'],
        ['It took time but it is progressing', 'positive'],
        ['We have to take everything in our mind for the exam, it is dangerous for our brain', 'negative'],
        ['Student are familiar with the current trend', 'positive'],
        ['Every department shoud be worked syncronizely', 'neutral'],
        ['In this last few decades, it is somehow satisfying the student', 'positive'],
    ]

    


    # Split the data into training and testing sets & extracting the testing_labels
    def split_data(data, test_ratio):
        """
        Split the input data into training and testing data sets & also extract 'labels' of each sentence of 'test_data'.
        :param data: List of tuples where each tuple represents a data point and its corresponding label.
        :param test_ratio: The ratio of data to be used for testing. The value must be between 0 and 1.
        :return: A tuple of two lists containing the training, testing data  & labels of each sentence of 'test_data' respectively.
        """
        random.shuffle(data)  # Shuffle the data randomly.
        split_index = int(len(data) * test_ratio)  # Compute the index to split the data.
        test_data = data[:split_index]  # Extract the testing data.
        train_data = data[split_index:]  # Extract the training data.
        test_labels = [label for (_, label) in test_data]  # Extract the labels for the testing data.
        test_data = [data_point for (data_point, _) in test_data]  # Extract the testing data points.

        return train_data, test_data, test_labels


    # Split the data into training and testing sets & extracting the testing_labels
    def test_data_only(data, test_ratio):
        
        print("\n\n%%%%%>> extracting only test_data <<%%%%%%")

        n = len(data)
        
        test_size = int(n * test_ratio)
        random.shuffle(data)                                                                   # here shuffle() function reorder the position of list element randomly

        return data[:test_size]        #-------->> The split is done by slicing the data array in two parts using Python's slice notation.

    # Preprocess the text data
    def preprocess_text(text, location):    
        print("===========>> preprocess() called from: {} <<============".format(location))
        print("  Before preprocessing:", text)

        text = text.lower() # Convert to lowercase
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'\d+', '', text) # Remove digits
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
        text = text.strip() # Remove leading and trailing whitespace
        print("  After preprocessing:", text)
        print("---------------|end of preprocess|--------------------\n")
        return text


    def preprocess_feedBackOnly(text):    

        text = text.lower() # Convert to lowercase
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'\d+', '', text) # Remove digits
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
        text = text.strip() # Remove leading and trailing whitespace
        
        return text




    # Compute the class prior probability of each class
    def compute_prior(data):
        print("%%%%%%%%%%%%%%>> I am compute_prior Function <<%%%%%%%%%%%%%%")
        n = len(data)

        positive_count = sum(1 for row in data if row[1] == 'positive')
        print("Total positive data= ",positive_count)
        negative_count = sum(1 for row in data if row[1] == 'negative')
        print("\nTotal negative data= ",negative_count)
        neutral_count = n - positive_count - negative_count
        print("\nTotal neutral data= ",neutral_count)
        print("-------------------|end of comput_prior|-------------------\n\n")
        return {'positive': positive_count / n, 'negative': negative_count / n, 'neutral': neutral_count / n}        #---> here we have key: value_pair so, its a dictionary.
    


        '''

        To calculate the proportion of each label type in the dataset, the count of each label type is divided by the total number of items in the dataset (n).
        This normalization is done to get the proportions of the different label types, which can then be used to analyze the distribution of the labels in the dataset.        

        '''


    # Compute the conditional probability of each word given each class
    def compute_likelihood(data):
        print("%%%%%%%%%%%%%%>> I am compute_likelihood Function <<%%%%%%%%%%%%%%")
        word_count = {'positive': {}, 'negative': {}, 'neutral': {}}           #--> here "word_count" is a dictionary because it has key & value pair
        word_total = {'positive': 0, 'negative': 0, 'neutral': 0}              #--> example "word_count": {'positive': {good, nice, like}, 'negative': {not, bad, theoritical}, 'neutral': {not bad, normal}}

        for ix, row in enumerate(data):                                                #  enumerate returns an enumerate object, which is an iterator that generates a series of tuples containing both
         
            print()
            #print(ix, row[0])
            # above print statement print as below i.e index & sentence
            # 0 I have Nepal education system.
            text = preprocess_text(row[0], "[ likelyhood() ]")
            text_label= row[1]

            words = text.split()
            print("Tokenize: ",words, ":", text_label)                 #-------> print data as:: ['nepal', 'education', 'system', 'is', 'not', 'so', 'bad']

            print("")

            for index, word in enumerate(words): 
                #print(word)

                if word not in word_count[row[1]]:
                    word_count[row[1]][word] = 0                                    #------>> As word count have "keys(either positive, negative or neutral)" and its  value "{}" and here "row[1]" give the class label and "[word]" gives a particular split word

                word_count[row[1]][word] += 1                                       #------->> weather if condition is valid or not but this statement execute so, if condition not satisfied then that particular word will be stored in respective Keys of word_count dictionary
                word_total[row[1]] += 1
                print("--->>", index, row[1], word_count[row[1]])

        for cls in ['positive', 'negative', 'neutral']:
            print("")
            print("")
            print("[",cls,"]", "class word length= ", word_total[cls])
            print("------------------------------")
            #------>> "cls" means class
            for word in word_count[cls]:
                word_freq= word_count[cls][word],
                
                #conditional_probability_of_word: cp_word
                cp_word=  word_count[cls][word]/word_total[cls]
                print("[",word,"]", "count= ",word_freq, "class[",word_total[cls],"]","word probability= ", cp_word)

            print("")
            
            
        
        print("----------------------|end of likelyhood()|----------------------------\n\n\n")
    
        return word_count






   

    # Compute the log posterior probability of each class given a feedback
    def predict(feedback, prior, called_from_msg, likelihood):
        print("%%%%%%%%%%%%%%>> I am main predict Function <<%%%%%%%%%%%%%%")
  
        feedback = preprocess_text(feedback, "[ predict() ]")
        words = feedback.split()
        
        #--> It initializes a dictionary called log_prob with the log prior probabilities of each class.
        log_prob = {'positive': log(prior['positive']), 'negative': log(prior['negative']), 'neutral': log(prior['neutral'])}  
        for word in words:
            if word in likelihood['positive']:
                #print(" found ", word, "=", likelihood)
                log_prob['positive'] += log(likelihood['positive'][word])
            else:
                #print("Not found ", word, "=", likelihood)
                log_prob['positive'] += log(1e-10)
            if word in likelihood['negative']:
                log_prob['negative'] += log(likelihood['negative'][word])
            else:
                log_prob['negative'] += log(1e-10)                              
            if word in likelihood['neutral']:                                   
                log_prob['neutral'] += log(likelihood['neutral'][word])         
            else:
                log_prob['neutral'] += log(1e-10)
         
        print("---------------|end of predict() |--------------------")      
        '''

            A small value like 1e-10 is used to solve the problem of zero probability.
            When a word in the feedback is not found in the training data, the corresponding             
            probability will be zero.

        '''  
        return max(log_prob, key=log_prob.get)

    

    """
        Notes about below "compute_confusion_matrix()"
        -----------------------------------------------
        
        
        i> classes = ['positive', 'negative', 'neutral']
           print(range(len(classes))

           what will be the output ?

           Ans: The output of the above code will be:
                range(0, 3)
                Here, len(classes) is 3, so range(len(classes)) returns an iterable object containing numbers from 0 up to (but not including) 3,
                which are 0, 1 and 2. This iterable object is passed to the range() function, which returns a range object that represents the sequence
                of numbers from 0 to 2. The print() function then prints this range object.

    
    
    """



  
    


    def compute_confusion_matrix(test_data, predicted_labels, actual_labels):
        print("\n\n================>> confusion_matrix() is called <<==================\n")
        classes = ['positive', 'negative', 'neutral']
        confusion_matrix = np.zeros((3, 3), dtype=int)


        print(predicted_labels)
        print(actual_labels)
        print("\n\n")
        
        for i in range(len(test_data)):
            predicted_class = predicted_labels[i]
            actual_class = actual_labels[i]
            predicted_index = classes.index(predicted_class)
            actual_index = classes.index(actual_class)
            confusion_matrix[actual_index][predicted_index] += 1


            print(confusion_matrix)
            print("\n")

            # Calculate total number of predictions
            total = sum(sum(confusion_matrix))  # ---> inner sum() function performs sum of all values in columns wise which convert our matrix to only one row with 3 dimensional element
                                                # and then the outer sum() function will perfom sumation on that single row to generate a single value. 
                                                # for reference visit: "sum().jpg": D:\seven Semester\Project\sentiment analysis\Final Final project\Notes



            # Calculate true positives, true negatives, false positives, and false negatives for each class
            tp = [confusion_matrix[i][i] for i in range(len(classes))]  # range(len(classes)) ---> generate the list of index of 
            
            # tn = [sum([confusion_matrix[j][k] for j in range(len(classes)) for k in range(len(classes)) if j != i and k != i]) for i in range(len(classes))]  #OR
            # ------>> calculating the "True Negative" [TN] <<------------                                                  
            tn = []
            for i in range(len(classes)):
                tn_i = 0
                for j in range(len(classes)):
                    for k in range(len(classes)):
                        if j != i and k != i:
                            tn_i += confusion_matrix[j][k]
                tn.append(tn_i)   
            # -------------------------------------------------------------      



            # fp = [sum([confusion_matrix[j][i] for j in range(len(classes))]) - tp[i] for i in range(len(classes))]    OR
            # ------>> calculating the "True Negative" [TN] <<------------                                                  
            fp = []
            for i in range(len(classes)):
                fp_count = 0
                for j in range(len(classes)):
                    if j != i:
                        fp_count += confusion_matrix[j][i]
                fp_count -= tp[i]
                fp.append(fp_count)
            # -------------------------------------------------------------          
            
            
            # fn = [sum([confusion_matrix[i][j] for j in range(len(classes))]) - tp[i] for i in range(len(classes))]          OR
            fn = []
            for i in range(len(classes)):
                sum_fn = 0
                for j in range(len(classes)):
                    if i != j:
                        sum_fn += confusion_matrix[i][j]
                fn.append(sum_fn)


            # Calculate accuracy, error rate, precision, and recall for each class
            accuracy = [tp[i] / total for i in range(len(classes))]
            error_rate = [1 - accuracy[i] for i in range(len(classes))]
            precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(len(classes))]
            recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(classes))]

            

        # print("\n\nAccuracy:", accuracy)
        # print("Error Rate:", error_rate)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("\n============>> end of confusion matrix <<=============\n\n")
        # print("\n\n")
        
        



    print("\n--------->> This registration() will be called twice.<<-----------")
    print("i> when form load for ist time.")
    print("ii> when we click on 'classify' button.")


    def compute_accuracy():
        # Make predictions on the test data
        test_data_ = test_data_only(data, 0.2)
        predicted_labels = []
        actual_labels = []
        
        for row in test_data_:
            feedback, actual_label = row
            predicted_label = predict(feedback, prior, "predict() called from accuracy value", likelihood)
            predicted_labels.append(predicted_label)
            actual_labels.append(actual_label)

        # Compute accuracy
        correct = sum(1 for i in range(len(test_data_)) if predicted_labels[i] == actual_labels[i])
        accuracy = correct / len(test_data_) * 100  # Corrected calculation, multiplied by 100 for percentage
        
        return accuracy

   



    if request.method == 'POST':
        
           

        opinion = request.POST.get('opinion')
        if opinion:
            feedback = opinion

            # spliting the data into training, testing  & extracting 'label' of "text"
            train_data, test_data, test_labels = split_data(data, 0.2)

            # -------->> printing the training data <<-------------------

            print("---------------->> Training Dataset <<------------------------")
            for a, b in enumerate(train_data):
                print(" index: ", a, "   input= ", b)

            print("---------------------------------------------------------------")

            print("--------------->> Testing data <<-----------------------------")
            for a, b in enumerate(test_data):
                print(" index: ", a, "   input= ", b)

            print("--------------------------------------------------------------")    
                   

            feedbackPreProcess= preprocess_feedBackOnly(feedback)
            



            
            # Train the model
            print("\n\n Here [compute_prior() is called.")
            prior = compute_prior(train_data)
            print("\n\n Here compute_likelihood() is called.")
            likelihood = compute_likelihood(train_data)

            print("---------------------->> printing frequency of each word in each class <<---------------------")
            for a, b in enumerate(likelihood.items()):
               print(":", a, ":", b)

            print("----------------------------------------------------------------------------------------------------------------------\n\n\n")   

            
            
            classification = predict(feedback, prior, "\npredict() from_classification", likelihood)
            print("\n\n\n")
            print("==================================================")
            print("|                 The given feedback             |")
            print("==================================================")
            print("    ", feedback,"= ", classification , "\n\n")
            print("\nAccuracy of model: {}\n\n".format(compute_accuracy()))


         


            
            predicted_labels = [predict(text, prior, "\npredict() for confusion matrix", likelihood) for text in test_data]
            compute_confusion_matrix(test_data, predicted_labels, test_labels)
           

            
        

            if classification == "positive":
                img_path = 'image/pos_img.jpg'

            elif classification == "negative":
                img_path = 'image/neg_img.jpg'

            elif classification == "neutral":
                img_path = 'image/neu_img.jpg'

            else:
                img_path = 'image/default.jpg'  # a default image for unknown classification


            feedback_tokenize= feedbackPreProcess.split()
            context = {'classification': classification, 'img_path': img_path, 'inputText': feedback, 'feedbackPreProcessText': feedbackPreProcess, 'feedback_tokenize': feedback_tokenize}
        

            
            return render(request, 'reg.html', context)
        
            
        

    return render(request, 'reg.html')
    

