model recommended_style {
  kind ode
  parameters { ka, cl, v, tlag, f_oral }
  covariates { wt@linear }
  states { depot, central }
  routes {
    iv -> central
    oral -> depot {
      lag = tlag
      bioavailability = f_oral
    }
  }
  derive {
    if wt > 0.0 {
      cl_i = cl * pow(wt / 70.0, 0.75)
    } else {
      cl_i = cl
    }
    v_i = v
    ke = cl_i / v_i
  }
  dynamics {
    ddt(depot) = -ka * depot
    ddt(central) = ka * depot - ke * central + rate(iv)
  }
  outputs {
    cp = central / v_i
  }
}