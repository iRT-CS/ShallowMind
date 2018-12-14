import math
def iterate(nnID,maxLayers,maxNodes):
    #get digits of the function
    #find number of digits
     numDigits = math.log10(nnID)
    #Because 10 has 2 digits
     if(numDigits%1 == 0):
         numDigits = int(numDigits + 1)
     else:
         numDigits = int(math.ceil(numDigits))
    #get digits into array(backwards)
     digits = []
     for i in range(numDigits):
         digit = (nnID//(10**i))%10
         digits.append(digit)
     digits.append(0)
    #Find carryover
     digitToIncrease = 0
     while((digitToIncrease<maxLayers)and(digits[digitToIncrease]>=maxNodes)):
         digitToIncrease = digitToIncrease + 1
     if(digitToIncrease == maxLayers):
         return -1
     for i in range(digitToIncrease):
         digits[i] = 1
     digits[digitToIncrease] = digits[digitToIncrease] + 1
     #recombine into a number
     newID = 0
     for i in range(len(digits)):
         newID += (10**i)*digits[i]
     return newID
