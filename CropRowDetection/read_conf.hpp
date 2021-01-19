#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;

string readConf(string input){

    string path;
    char compare;

    if(input == "input"){
        compare = 'i';
    }
    else if(input == "sample_ExG"){
        compare = 'e';
    }
    else if(input == "sample_squeeze_unet"){
        compare = 's';
    }
    else{
        compare = 'g';
    }
    
    ifstream read_conf ("../config.ini");
    if (read_conf.is_open())
    {
        string line;
        while(getline(read_conf, line)){
            line.erase(remove_if(line.begin(), line.end(), ::isspace),
                                    line.end());
            if(line[0] == '[' || line.empty() || line[8] != compare)
                continue;
            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            path = value;
        }
    }
    else {
        cerr << "Couldn't open config file.\n";
    }

    return path;
}