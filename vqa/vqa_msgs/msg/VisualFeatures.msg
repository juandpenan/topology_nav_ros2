# This message contains the feature extracted from the MDTR model
std_msgs/Header header
FeaturesInfo info
float32[] acc #The accuracy values of the strings outputs in the same order as data
string[] data #The string outputs in the order of each question
