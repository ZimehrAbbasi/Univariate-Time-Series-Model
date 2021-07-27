import sys

class Domain:

    class Bucket:

        def __init__(self, size_cap):
            self.values = []
            self.curr_size = 0
            self.size_cap = size_cap
            self.next = None

        def append(self, value):

            if self.size_cap == self.curr_size:
                return False

            else:

                self.values.append(value)
                self.curr_size += 1
                return True

        def __str__(self):

            return str(self.values)

    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
        self.current_ticker = -1
        self.index = {}
        self.head = None
        self.size = 0

    def create_bucket(self):

        if self.bucket_size <= 0:
            self.bucket_size = 1
        elif self.bucket_size >= sys.maxsize:

            return

        bucket = self.Bucket(self.bucket_size)
        self.index[self.current_ticker] = bucket

        return bucket

    def iterate(self, arg=None, function=None):

        bucket = self.head

        i = 0
        while bucket != None:
            i += 1
            if function != None:
                function(bucket, arg)
            bucket = bucket.next

        self.current_ticker = i-1

    def add_value(self, value):

        if self.current_ticker == -1:
            self.current_ticker += 1
            bucket = self.create_bucket()
            self.index[self.current_ticker] = bucket
            bucket.append(value)
            self.head = bucket
            self.size += 1
            return

        try:
            curr_bucket = self.index[self.current_ticker]
        except:
            self.iterate()

        if not curr_bucket.append(value):
            self.current_ticker += 1
            bucket = self.create_bucket()
            self.index[self.current_ticker] = bucket
            self.index[self.current_ticker-1].next = bucket
            bucket.append(value)
            self.size += 1
