def printer(var1):
    
    print("I print " + var1)

def main():

    var_in_1 = "Hello World"
    var_in_2 = "1"
    var_in_3 = 1
    printer(var_in_1) # I print Hello World
    printer(var_in_2) # I print 1
    printer(var_in_3) # TypeError: can only concatenate str (not "int") to str

if __name__ == '__main__':
    main()

