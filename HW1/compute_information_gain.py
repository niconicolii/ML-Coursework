import math

def compute_information_gain(keyword):
    data = load_data()
    entropy = - 1968 / 3266 * math.log2(1968 / 3266) - 1298 / 3266 * math.log2(1298 / 3266)
    
    p_real = real_sample / (real_sample + fake_sample)
    p_fake = fake_sample / (real_sample + fake_sample)
    entropy_given_x = - p_real * math.log2(p_real) - p_fake * math.log2(p_fake)
    IG = entropy - entropy_given_x
    print("Information Gain in spliting at keyword '", keyword, "' with 1556 samples is ", IG)
    return IG

if __name__ == "__main__":
    compute_information_gain("hillary", 0.996, 1471,85)