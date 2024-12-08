class Hung:
    def __init__(self):
        self.__name = "Hung"
        self.age = 20

    def say_hello(self):
        print("Hello, my name is", self.__name)

    def say_age(self):
        print("I'm", self.age, "years old")


hung = Hung()
hung.say_hello()
hung.say_age()
