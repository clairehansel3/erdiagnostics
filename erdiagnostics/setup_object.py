# -*- coding: utf-8 -*-
class SetupObject(object):
    """
    A SetupObject is an object which checks that if an attributed in the list
    'attributes_requiring_resetup' is changed, the method 'setup()' is called
    before a method decorated with @ensure_setup is run. Note: when overloading
    'setup()' in subclasses make sure to call SetupObject's 'setup()' either
    directly or with 'super()'. Also note that if you write
    'attributes_requiring_resetup = list_1' in BaseClass and then in SubClass
    write 'attributes_requiring_resetup = list_2', list_2 will overwrite list_1.
    What you should instead do in SubClass is write
    'attributes_requiring_resetup = BaseClass.attributes_requiring_resetup +
    list_2'.
    """

    attributes_requiring_resetup = []

    def __init__(self):
        self._is_setup = False

    def __setattr__(self, name, value):
        if name in self.attributes_requiring_resetup:
            self._is_setup = False
        super(SetupObject, self).__setattr__(name, value)

    def setup(self):
        self._is_setup = True

def ensure_setup(method):
    """
    When this is used to decorate a method of a subclass of SetupObject, it
    checks that the object is set up every time that method is called. If the
    object is not set up, an exception is thrown.
    """
    def wrapper(self, *args, **kwargs):
        if not self._is_setup:
            raise Exception('you forgot to call .setup()')
        return method(self, *args, **kwargs)
    return wrapper
