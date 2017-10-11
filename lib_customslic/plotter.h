#if !defined(_PLOTTER_H_INCLUDED_)
#define _PLOTTER_H_INCLUDED_

#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <thread>
#include <chrono>

#include "./gnuplot-iostream/gnuplot-iostream.h"

using namespace std;

class DataSeries
{
public:
  string title;
  vector<double> data;
  vector<std::pair<double, double>> data_pair;
  int max_points;

  string type_string;
  string initial_string;

  DataSeries (string initial_string, string title, string type_string, int max_points)
  {
    this->max_points = max_points;
    this->title = title;
    this->type_string = type_string;
    this->initial_string = initial_string;
  }
  DataSeries ()
  {
    max_points = 0;
  }

  void add_point (double val)
  {
    if (data.size () > max_points)
      data.erase (data.begin ());

    data.push_back (val);
  }

  void add_point (double x, double y)
  {
    if (data_pair.size () > max_points)
      data_pair.erase (data_pair.begin ());

    data_pair.push_back (std::make_pair (x, y));
  }

  virtual void do_plot (Gnuplot & gp)
  {
      int data_size = data.size ();
      int data_pair_size = data_pair.size ();
      if (data_size == 0 && data_pair_size == 0)
        return;

      gp << initial_string << " '-' binary ";

      if (data_size > 0)
		    gp << gp.binFmt1d(data, "array");
      else
		    gp << gp.binFmt1d(data_pair, "record");

      gp << " with " << type_string << " title \"" << title << "\" ";
  }

  virtual void print_binary_data (Gnuplot & gp)
  {
    int data_size = data.size ();
    int data_pair_size = data_pair.size ();
    if (data_size == 0 && data_pair_size == 0)
      return;

    if (data_size > 0)
      gp.sendBinary1d(data);
    else
      gp.sendBinary1d(data_pair);
  }

};

class Plot
{
public:
  string title;
  std::map<string, DataSeries> series;

  string pre_plot_cmds;
  string post_plot_cmds;

  Plot (string title) : title (title)
  { }

  Plot () { }

  void do_plot (Gnuplot & gp)
  {
     if (series.size () == 0)
        return;

    gp << "set title \"" << title << "\" \n;";

    gp << pre_plot_cmds << " \n";

    gp << "plot ";

    int count = series.size ();
    for (auto& s:series)
    {
		   s.second.do_plot (gp);

       if (count-- > 1)
          gp << ", ";
    }
    gp << " \n";

    for (auto& s:series)
		  s.second.print_binary_data (gp);

    gp << post_plot_cmds << " \n";
  }
};

class Plotter
{
public:
	Gnuplot gp;
  std::map<string, Plot> plots;
  
  string pre_plot_cmds;
  string post_plot_cmds;

  float fps;
  std::chrono::time_point<std::chrono::system_clock> last_refresh;

  Plotter (float fps) : fps (fps)
  {
    last_refresh = std::chrono::system_clock::now ();
  }

  ~Plotter()
  {
    gp << "exit\n";
		gp.flush();
  }

	void reset ();

  bool should_plot ()
  {
    auto current = std::chrono::system_clock::now ();
    auto diff = current - last_refresh;
    long nano_sec = diff.count ();
    bool ret = (nano_sec/1E9) > 1/fps;
    if (ret)
      last_refresh = current;
    return ret;
  }

	void do_plot ()
  {
    if (!should_plot ())
      return;

    gp << pre_plot_cmds << " \n";

    for (auto& p:plots)
		  p.second.do_plot (gp);

    gp << post_plot_cmds << " \n";

		gp.flush();
  }
};

#endif // !defined(_PLOTTER_H_INCLUDED_)
