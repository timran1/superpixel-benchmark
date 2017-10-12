#include <iostream>
#include <cmath>
using namespace std;

class fixedp {

public:
    unsigned int integer;
    unsigned int fraction;


    unsigned long long value;
    fixedp convertTo(const int& integer);
    fixedp convertTo(const float& flt);
    int   convertTo(const fixedp& fix);

    float convertToflt(const fixedp& fixp)
    {
        return ((float)(fixp.value) / (1ull << fixp.fraction));
    }

    double convertToDouble(const fixedp& fixp)
    {
        return ((double)(fixp.value) / (1ull << fixp.fraction));
    }

    fixedp& operator=(const fixedp& other) // copy assignment
    {
        double temp;
        /* this has already defined precision and range */
        if(this!= &other)
        {
            //temp = convertToDouble(other);
        	int diff = other.fraction - fraction;
        	if (diff <= 0)
        		this->value = other.value << (-diff);
			else
				this->value = other.value >> (diff);


//            this->value = (temp) * (1ULL << this->fraction);
//            this->value += (this->value >= 0) ? 0.5f : -0.5f;
        }

        return *this;
    }

    fixedp& operator=(const int& other) // copy assignment
    {
        this->convertTo(other);
        return *this;
    }
    fixedp operator*(const fixedp& rhs)
    {
    	fixedp out;
		out.value = (rhs.value * this->value);
        
    	out.integer = this->integer + rhs.integer +1;
    	out.fraction = this->fraction + rhs.fraction +2;

        return out;
    }

    fixedp operator+(const fixedp& rhs)
    {
    	fixedp out;
    	out.value = (rhs.value + this->value);

    	out.integer = max(this->integer ,rhs.integer)+1;
    	out.fraction = max(this->fraction, rhs.fraction);


        return out;
    }

    fixedp operator+=(const fixedp& rhs)
    {
        (*this) = (*this) + rhs;
        return (*this);
    }

    fixedp operator-(const fixedp& rhs)
    {
    	fixedp out;
    	out.value = (this->value - rhs.value);

    	out.integer = max(this->integer ,rhs.integer)+1;
    	out.fraction = max(this->fraction, rhs.fraction);


        return out;
    }

    fixedp operator/(const fixedp& rhs)
    {
    	fixedp out;
    	out.value = (rhs.value / this->value);

    	out.fraction = this->fraction- rhs.fraction;


        return out;
    }

    bool operator<(const fixedp& rhs)
    {
        float left, right;
        right = convertToflt(rhs);
        left = convertToflt(*this);

        return (left < right);
    }

    bool operator<=(const fixedp& rhs)
    {
        float left, right;
        right = convertToflt(rhs);
        left = convertToflt(*this);

        return (left <= right);
    }

    fixedp()
    {
        this->value=this->integer=this->fraction = 0;
    }

    operator int()   { return convertTo(*this);}
    operator float() { return convertToflt(*this);}

    fixedp(unsigned int integer, unsigned fraction, const int& value);
    fixedp(unsigned int integer, unsigned fraction, const float& value);
};

fixedp fixedp::convertTo(const int& integer)
{
    this->value = (integer) * (1ull << this->fraction);
    return *this;
}


int fixedp::convertTo(const fixedp& fixp)
{
    return ((fixp.value) / (1ull << fixp.fraction));
}


fixedp fixedp::convertTo(const float& flt)
{
	if (flt == FLT_MAX)
	{
		//this->value = ((1 << (integer+1))-1);
		this->value = -1;
	}else
	{
	    this->value = (flt) * (1ULL << this->fraction);

	    this->value += (this->value >= 0) ? 0.5f : -0.5f;
	}

    return *this;
}

fixedp::fixedp(unsigned int integer, unsigned fraction, const int& value)
{
    this->fraction = fraction;
    this->integer = integer;
    *this= convertTo(value);

}

fixedp::fixedp(unsigned int integer, unsigned fraction, const float& value)
{
    this->fraction = fraction;
    this->integer = integer;
    *this = convertTo(value);

}
