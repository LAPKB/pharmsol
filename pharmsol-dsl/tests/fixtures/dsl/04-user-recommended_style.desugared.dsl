model recommended_style {
  kind ode
  parameters {
    ka,
    ke,
    v,
  }
  states {
    depot,
    central,
  }
  routes {
    oral -> depot
  }
  dynamics {
    ddt(depot) = -ka * depot
    ddt(central) = ka * depot - ke * central
  }
  outputs {
    cp = central / v
  }
}