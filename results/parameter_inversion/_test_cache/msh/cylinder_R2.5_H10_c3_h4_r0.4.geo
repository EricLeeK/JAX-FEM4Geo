
        Point(1) = {0, 0, 0, 1.0};
        Point(2) = {-1.0, 1.0, 0, 1.0};
        Point(3) = {1.0, 1.0, 0, 1.0};
        Point(4) = {1.0, -1.0, 0, 1.0};
        Point(5) = {-1.0, -1.0, 0, 1.0};
        Point(6) = {2.5*Cos(3*Pi/4), 2.5*Sin(3*Pi/4), 0, 1.0};
        Point(7) = {2.5*Cos(Pi/4), 2.5*Sin(Pi/4), 0, 1.0};
        Point(8) = {2.5*Cos(-Pi/4), 2.5*Sin(-Pi/4), 0, 1.0};
        Point(9) = {2.5*Cos(-3*Pi/4), 2.5*Sin(-3*Pi/4), 0, 1.0};

        Line(1) = {2, 3};
        Line(2) = {3, 4};
        Line(3) = {4, 5};
        Line(4) = {5, 2};
        Line(5) = {2, 6};
        Line(6) = {3, 7};
        Line(7) = {4, 8};
        Line(8) = {5, 9};

        Circle(9) = {6, 1, 7};
        Circle(10) = {7, 1, 8};
        Circle(11) = {8, 1, 9};
        Circle(12) = {9, 1, 6};

        Curve Loop(1) = {1, 2, 3, 4};
        Plane Surface(1) = {1};
        Curve Loop(2) = {1, 6, -9, -5};
        Plane Surface(2) = {2};
        Curve Loop(3) = {2, 7, -10, -6};
        Plane Surface(3) = {3};
        Curve Loop(4) = {3, 8, -11, -7};
        Plane Surface(4) = {4};
        Curve Loop(5) = {4, 5, -12, -8};
        Plane Surface(5) = {5};

        Transfinite Curve {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} = 3 Using Progression 1;

        Transfinite Surface {1};
        Transfinite Surface {2};
        Transfinite Surface {3};
        Transfinite Surface {4};
        Transfinite Surface {5};
        Recombine Surface {1, 2, 3, 4, 5};

        Extrude {0, 0, 10} {
          Surface{1:5}; Layers {4}; Recombine;
        }

        Mesh 3;