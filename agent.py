class Agent:
    def __init__(self):
        self.__name = "Agent"
        self.age = 30

    def say_hello(self):
        print("Hello, my name is", self.__name)

    def change_name_Hung(self, hung):
        hung._Hung__name = "Huy"
        hung.age = 21
